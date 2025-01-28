use outlines_core::prelude::*;

fn main() {
    let schema = std::io::read_to_string(std::io::stdin()).unwrap();
    let regex = json_schema::regex_from_str(&schema, None).unwrap();
    println!("Regex: {}", regex);
    println!("Regex len: {}", regex.len());
}
