use std::io::{self, Error};
use std::fs::{self, DirEntry};
use std::path::Path;
use std::{error, fmt};
use std::result;
use std::f64::consts::PI;

use opencv::prelude::HOGDescriptorTraitConst;
use opencv::{
    core::*,
    Result,
    imgcodecs,
    imgproc,
    objdetect::{
        HOGDescriptor,
        HOGDescriptor_HistogramNormType,
        HOGDescriptor_DEFAULT_NLEVELS,
    },
    ml::prelude::*,
    ml::SVM,
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

fn compute_hog(path_str: &str, output: &mut Vec<Vector<f32>>) -> Result<()> {
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
        output.push(descriptor);
    }
    Ok(())
}

fn main() {

}
