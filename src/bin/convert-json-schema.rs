use outlines_core::prelude::*;

fn print_help() {
    println!("JSON Schema to Regex Converter\n");
    println!("Usage:");
    println!("  cat schema.json | convert-json-schema");
    println!("  convert-json-schema --help\n");
    println!("Options:");
    println!("  --help    Show this help message\n");
    println!("Description:");
    println!("  Reads a JSON Schema from stdin and converts it to a regular expression.");
}

fn main() {
    if std::env::args().any(|arg| arg == "--help") {
        print_help();
        return;
    }

    let schema = std::io::read_to_string(std::io::stdin()).unwrap();
    let regex = json_schema::regex_from_str(&schema, None, None).unwrap();
    println!("Regex: {}", regex);
    println!("Regex len: {}", regex.len());
}
