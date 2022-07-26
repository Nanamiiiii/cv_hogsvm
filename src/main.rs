use std::io::{self, Error};
use std::fs::{self, DirEntry};
use std::path::Path;
use std::{error, fmt};
use opencv::prelude::*;
use opencv::ml::prelude::*;
use opencv::ml::SVM;

type Result<T> = std::result::Result<T, ModelCreationError>;

#[derive(Debug, Clone)]
struct ModelCreationError;

impl fmt::Display for ModelCreationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "cannot create svm model. check the resource data.")
    }
}

fn get_files(path_str: &str) -> Result<Vec<String>> {
    let path = Path::new(path_str);
    if !path.is_dir() {
        return Err(ModelCreationError)
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
