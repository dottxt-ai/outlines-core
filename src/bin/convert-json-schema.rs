use outlines_core::json_schema::build_regex_from_schema;

fn main() {
    let schema = std::io::read_to_string(std::io::stdin()).unwrap();
    let regex = build_regex_from_schema(&schema, None).unwrap();
    println!("Regex: {}", regex);
    println!("Regex len: {}", regex.len());
}
