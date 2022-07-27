use std::io::{self, Error};
use std::fs::{self, DirEntry};
use std::path::Path;
use std::{error, fmt};
use std::result;
use opencv::prelude::*;
use opencv::ml::prelude::*;
use opencv::ml::SVM;


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

fn create_svm_model(path_str: &str) {
    let train_filenames = match get_files(path_str) {
        Ok(vec) => vec,
        Err(e) => panic!("Error: {}", e) 
    };
    
}

fn main() {

}
