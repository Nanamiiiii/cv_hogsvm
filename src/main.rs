use std::fs;
use std::path::Path;
use std::{error, fmt};
use std::result;
use std::mem::size_of_val;
use std::os::raw::c_void;

use clap::Parser;

use opencv::ml::SVMConst;
use opencv::{
    core::*,
    Result,
    imgcodecs,
    imgproc,
    imgproc::LINE_8,
    types::*,
    prelude::HOGDescriptorTraitConst,
    objdetect::{
        HOGDescriptor,
        HOGDescriptor_HistogramNormType,
        HOGDescriptor_DEFAULT_NLEVELS,
        prelude::HOGDescriptorTrait,
    },
    ml::{
        SVM,
        SVM_Types,
        SVM_KernelTypes,
        StatModel,
        ROW_SAMPLE
    }
};


#[derive(Debug)]
enum ModelCreationError {
    InvalidPath,
    NoValidFile,
}

impl fmt::Display for ModelCreationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ModelCreationError::InvalidPath => write!(f, "invalid file path."),
            ModelCreationError::NoValidFile => write!(f, "no valid files in given directory."),
        }
    }
}

impl error::Error for ModelCreationError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            ModelCreationError::InvalidPath => None,
            ModelCreationError::NoValidFile => None,
        }
    }
}

fn get_files(path_str: &str) -> result::Result<Vec<String>, ModelCreationError> {
    let path = Path::new(path_str);
    if !path.is_dir() {
        return Err(ModelCreationError::InvalidPath)
    }
    let mut files: Vec<String> = Vec::new();
    for entry in fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        let filepath = entry.path();
        if filepath.is_dir() {
            continue;
        }
        files.push(filepath.to_str().unwrap().to_owned())
    }
    if files.is_empty() {
        return Err(ModelCreationError::NoValidFile)
    }
    Ok(files)
}

fn compute_hog(path_str: &str, output: &mut VectorOfMat) -> Result<()> {
    /* Get images' path from directory */
    let train_filenames = match get_files(path_str) {
        Ok(vec) => vec,
        Err(e) => panic!("Error: {}", e), 
    };

    /* Processing images */
    for imgfile in train_filenames {
        /* Load images */
        let img = imgcodecs::imread(&imgfile, imgcodecs::IMREAD_COLOR)?;

        /* Grayscale */
        let mut gray_scale = Mat::default();
        imgproc::cvt_color(
            &img,
            &mut gray_scale,
            imgproc::COLOR_BGR2GRAY, 
            0
        )?;

        /* Downscale to 64x64 */
        let mut downscale = Mat::default();
        imgproc::resize(&gray_scale, 
            &mut downscale, 
            Size::new(64, 64), 
            0.0, 
            0.0, 
            imgproc::INTER_LINEAR
        )?;

        let windows_sz = Size::new(64, 64);
        let block_sz = Size::new(16, 16);
        let block_str = Size::new(4, 4);
        let cell_sz = Size::new(4, 4);
        let nbins = 9;
        let hog = HOGDescriptor::new(
            windows_sz, 
            block_sz, 
            block_str, 
            cell_sz,
            nbins,
            1,
            -1.0,
            HOGDescriptor_HistogramNormType::L2Hys,
            0.2,
            false,
            HOGDescriptor_DEFAULT_NLEVELS,
            false
        )?; 
        let mut descriptor: Vector<f32> = Vector::new();
        hog.compute(&downscale, 
            &mut descriptor, 
            Size::default(), 
            Size::default(), 
            &Vector::new()
        )?;
        output.push(Mat::from_exact_iter(descriptor.into_iter())?);
    }
    Ok(())
}

fn convert_train_set(train_data: VectorOfMat, output: &mut Mat) -> Result<()> {
    let rows: i32 = train_data.len() as i32;
    let cols: i32 = train_data.get(0)?.cols();
    let mut tmp = unsafe {
        Mat::new_rows_cols(1, cols, CV_32FC1)?
    };
    unsafe {
        output.create_rows_cols(rows, cols, CV_32FC1)?
    }

    for i in 0..train_data.len() {
        assert!(train_data.get(i)?.cols() == 1 || train_data.get(i)?.rows() == 1);
        if train_data.get(i)?.cols() == 1 {
            transpose(&train_data.get(i)?, &mut tmp)?;
            tmp.copy_to(&mut output.row(i as i32)?)?;
        } else if train_data.get(i)?.rows() == 1 {
            train_data.get(i)?.copy_to(&mut output.row(i as i32)?)?;
        }
    }
    Ok(())
}

fn svm_train(positive: VectorOfMat, negative: VectorOfMat, output_file: &str) -> Result<()> {
    let label_positive = vec![0; positive.len()];
    let label_negative = vec![1; negative.len()];
    let labels: VectorOfi32 = VectorOfi32::from_iter(
        label_positive
        .into_iter()
        .chain(label_negative.into_iter())
        .into_iter()
    );
    
    let train_data: VectorOfMat = positive
        .into_iter()
        .chain(negative.into_iter())
        .collect();
    
    let mut train_set = Mat::default();
    convert_train_set(train_data, &mut train_set)?;
    
    let mut svm = <dyn SVM>::create()?;
    svm.set_type(SVM_Types::C_SVC as i32)?;
    svm.set_kernel(SVM_KernelTypes::LINEAR as i32)?;
    svm.set_gamma(1.0)?;
    svm.set_c(1.0)?;
    svm.set_term_criteria(TermCriteria {
        typ: TermCriteria_Type::COUNT as i32,
        max_count: 100,
        epsilon: 1.0e-6
    })?;
    svm.train(&train_set, ROW_SAMPLE, &labels)?;
    svm.save(output_file)?;
    Ok(())
}

fn get_svm_detector(svm: Ptr<dyn SVM>) -> Result<VectorOff32> {
    let sv = svm.get_support_vectors()?;
    let sv_num = sv.rows();
    let mut alpha = Mat::default();
    let mut svidx = Mat::default();
    let rho = svm.get_decision_function(0, &mut alpha, &mut svidx)?;

    assert!(alpha.total() == 1 && svidx.total() == 1 && sv_num == 1);
    assert!(
        (alpha.typ() == CV_64F && *alpha.at::<f64>(0)? == 1.0) ||
        (alpha.typ() == CV_32F && *alpha.at::<f32>(0)? == 1.0)
    );
    assert!(sv.typ() == CV_32F);

    let mut hog_detector = VectorOff32::with_capacity(sv.cols() as usize + 1);
    unsafe {
        hog_detector.as_raw_mut_VectorOff32().copy_from(sv.ptr(0)? as *const c_void, sv.cols() as usize * size_of_val(&hog_detector.get(0)?));
    }
    hog_detector.set(sv.cols() as usize, -rho as f32)?;
    Ok(hog_detector)
}

fn create_hog_detector(svm_filename: &str, detector_filename: &str) -> Result<()> {
    let svm = <dyn SVM>::load(svm_filename)?;
    let mut hog = HOGDescriptor::default()?;
    hog.set_win_size(Size::new(64, 64));
    hog.set_svm_detector(&get_svm_detector(svm)?)?;
    hog.save(detector_filename, "svm")?;
    Ok(())
}

fn detect_hog_multiscale(target_path: &str, hog_detector: &str, result_path: &str) -> Result<()> {
    let mut hog = HOGDescriptor::default()?;
    hog.load(hog_detector, "svm")?;

    let target_files = match get_files(target_path) {
        Ok(vec) => vec,
        Err(e) => panic!("{}", e)
    };

    for file in target_files {
        let mut img = imgcodecs::imread(&file, imgcodecs::IMREAD_COLOR)?;
        let mut detections = VectorOfRect::new();
        let mut found_weights = VectorOff64::new();
        hog.detect_multi_scale_weights(
            &img, 
            &mut detections, 
            &mut found_weights,
            0.0, 
            Size::default(), 
            Size::default(), 
            1.05, 
            2.0, 
            false
        )?;

        for i in 0..detections.len() {
            let color = Scalar::new(
                0.0, 
                found_weights.get(i)? * found_weights.get(i)? * 200.0, 
                0.0,
                0.0
            );
            imgproc::rectangle(&mut img, 
                detections.get(i)?, 
                color, 
                1, 
                LINE_8, 
                0
            )?;
            let out = result_path.to_owned() + &i.to_string() + ".png";
            imgcodecs::imwrite(&out, &img, &VectorOfi32::new())?;
        }
    }
    Ok(())
}

#[derive(Parser)]
#[clap(
    name = "cv_hogsvm",
    author = "Akihiro Saiki",
    version = "0.1.0",
    about = "Obj detection using hog & svm"
)]
struct AppArgs {
    #[clap(short = 's', long = "svm")]
    svm_train: bool,
    #[clap(short = 'h', long = "hog")]
    hog_detector: bool,
    #[clap(short = 'd', long = "detect")]
    detect: bool,
}

fn main() {
    let args: AppArgs = AppArgs::parse();

    let svm_train_flag = args.svm_train;
    let hog_detector = args.hog_detector;
    let detection = args.detect;

    let svm_filename = "resource/svm_traindata.xml";
    let detector_filename = "resource/hog_svm_detector.yml";
    let positive_samples = "resource/base";
    let negative_samples = "resource/negative";
    let detect_target = "resource/target";
    let detection_result = "resource/result";

    if svm_train_flag {
        let mut positive_hogs = VectorOfMat::new();
        let mut negative_hogs = VectorOfMat::new();
        compute_hog(positive_samples, &mut positive_hogs).unwrap();
        compute_hog(negative_samples, &mut negative_hogs).unwrap();
        svm_train(positive_hogs, negative_hogs, svm_filename).unwrap();
    }

    if hog_detector {
        create_hog_detector(svm_filename, detector_filename).unwrap();
    }

    if detection {
        detect_hog_multiscale(
            detect_target, 
            detector_filename, 
            detection_result
        ).unwrap();
    }
}
