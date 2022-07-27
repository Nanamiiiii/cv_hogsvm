use std::io::{self, Error};
use std::fs::{self, DirEntry};
use std::path::Path;
use std::{error, fmt};
use std::result;
use opencv::{
    core::*,
    Result,
    prelude::*,
    imgcodecs,
    imgproc,
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

fn create_svm_model(path_str: &str) -> Result<()> {
    /* Get images' path from directory */
    let train_filenames = match get_files(path_str) {
        Ok(vec) => vec,
        Err(e) => panic!("Error: {}", e), 
    };

    /* Processing images */
    for imgfile in train_filenames {
        /* Load images */
        let img = imgcodecs::imread(&imgfile, imgcodecs::IMREAD_COLOR).unwrap();

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

        /* Apply sobel filter 
         * to get edge image
         */
        let mut edge_x = Mat::default();
        let mut edge_y = Mat::default();
        imgproc::sobel(
            &downscale,
            &mut edge_x,
            CV_32F,
            1,
            0,
            3,
            1.0,
            0.0,
            BORDER_DEFAULT
        )?;
        imgproc::sobel(
            &downscale,
            &mut edge_y,
            CV_32F,
            0,
            1,
            3,
            1.0,
            0.0,
            BORDER_DEFAULT
        )?;

        /* Calc magnitude and angle of edge gradient */
        let mut magnitude = Mat::default();
        let mut angle = Mat::default();
        cart_to_polar(
            &edge_x,
            &edge_y,
            &mut magnitude,
            &mut angle,
            false
        )?;

        /* Quantization */

    } 
    Ok(())
}

fn main() {

}
