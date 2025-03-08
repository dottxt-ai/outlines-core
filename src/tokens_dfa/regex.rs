use regex_automata::util::alphabet::ByteClasses;
use regex_syntax::hir::Hir;
use regex_syntax::hir::HirKind;
use regex_syntax::hir::Class;
use rustc_hash::{FxHashSet as HashSet, FxHashMap as HashMap};
use std::collections::BTreeSet;

pub fn compile_dead_byte_classes(regex: &str, byte_classes: &ByteClasses) -> HashSet<u8> {
    let dead_bytes = static_dead_bytes(regex);
    let mut dead_bytes_set: HashSet<u8> = HashSet::default();

    for byte in dead_bytes{
        let class = byte_classes.get(byte);
        dead_bytes_set.insert(class);
    }
    dead_bytes_set
}

fn static_dead_bytes(pattern: &str) -> Vec<u8> {

    let mut builder = regex_syntax::ParserBuilder::new();
    builder.unicode(false);  
    builder.utf8(false);     
    
    let mut parser = builder.build();
    
    // Parser le pattern en HIR
    let hir = match parser.parse(pattern) {
        Ok(hir) => hir,
        Err(e) => {
            panic!("tokens_dfa::utils::static_dead_bytes: Parsing Error {:?}", e);
        }
    };

    
    let mut live_bytes = BTreeSet::new();
    collect_live_bytes(&hir, &mut live_bytes);

    
    let mut dead_bytes = Vec::new();
    for byte in 0..=255 {
        if !live_bytes.contains(&byte) {
            dead_bytes.push(byte);
        }
    }
    dead_bytes
}

fn collect_live_bytes(hir: &Hir, live_bytes: &mut BTreeSet<u8>) {
    match hir.kind() {
       
        HirKind::Literal(lit) => {
            for &byte in lit.clone().0.into_vec().iter() {
                live_bytes.insert(byte);
            }
        }
        // Char Class (ex: [a-z], [\x19-<])
        HirKind::Class(class) => {
            match class {
                Class::Bytes(byte_class) => {
                    for range in byte_class.ranges() {
                        for byte in range.start()..=range.end() {
                            live_bytes.insert(byte);
                        }
                    }
                }
                Class::Unicode(unicode_class) => {
                    // Unicode Class t
                    for range in unicode_class.ranges() {
                        // Convert every point code to UTF-8
                        for cp in range.start()..=range.end() {
                            let mut buf = [0u8; 4];
                            if let Some(c) = std::char::from_u32(cp as u32) {
                                let utf8_len = c.encode_utf8(&mut buf).len();
                                for &byte in &buf[..utf8_len] {
                                    live_bytes.insert(byte);
                                }
                            }
                        }
                    }
                }
            }
        }
        // Concat(ex: ab)
        HirKind::Concat(hirs) => {
            for h in hirs {
                collect_live_bytes(h, live_bytes);
            }
        }
        // Alternation (ex: a|b)
        HirKind::Alternation(hirs) => {
            for h in hirs {
                collect_live_bytes(h, live_bytes);
            }
        }
        // repetition (ex: a*)
        HirKind::Repetition(rep) => {
            collect_live_bytes(&rep.sub, live_bytes);
        }
        // Assertions (ex: ^, $, \b)
        HirKind::Look(look) => {

            if let regex_syntax::hir::Look::End = look{
                // $ n'ajoute pas de bytes spécifiques
                // Ne rien faire ici
            } else {
                match look {
                    regex_syntax::hir::Look::EndLF => {
                        live_bytes.insert(b'\n');
                    }
                    regex_syntax::hir::Look::EndCRLF => {
                        live_bytes.insert(b'\r');
                        live_bytes.insert(b'\n');
                    }
                    // Assertions de début de ligne
                    regex_syntax::hir::Look::StartLF => {
                        live_bytes.insert(b'\n');
                    }
                    regex_syntax::hir::Look::StartCRLF => {
                        live_bytes.insert(b'\r');
                        live_bytes.insert(b'\n');
                    }
                    // Limites de mots ASCII
                    regex_syntax::hir::Look::WordAscii |
                    regex_syntax::hir::Look::WordAsciiNegate |
                    regex_syntax::hir::Look::WordStartAscii |
                    regex_syntax::hir::Look::WordEndAscii |
                    regex_syntax::hir::Look::WordStartHalfAscii |
                    regex_syntax::hir::Look::WordEndHalfAscii => {
                        // Caractères de mot ASCII
                        for byte in b'a'..=b'z' {
                            live_bytes.insert(byte);
                        }
                        for byte in b'A'..=b'Z' {
                            live_bytes.insert(byte);
                        }
                        for byte in b'0'..=b'9' {
                            live_bytes.insert(byte);
                        }
                        live_bytes.insert(b'_');
                    }
                    _ => {
                        
                    }
                }
            }
            
            
        }
        // Capture
        HirKind::Capture(group) => {
            collect_live_bytes(&group.sub, live_bytes);
        }
        // Other Cases
        HirKind::Empty => { 
            
         }
        
    }
}

fn add_litteral( literals: &mut HashMap<String, Vec<usize>>, literal: String, pos:usize){
        literals.entry(literal).or_insert_with(Vec::default).push(pos);
}

pub fn extract_literals(regex: &str) -> HashMap<String, Vec<usize>> {
    let mut literals: HashMap<String, Vec<usize>> = HashMap::default();
    let mut buffer = String::new();
    let mut start_pos = None;
    let mut inside_brackets = false;
    let mut inside_parenthesis: bool = false;
    let mut prev_is_optional = false;
    let mut inside_escape: bool = false;
    let mut count_escape : u32 = 0;


    for (i, c) in regex.chars().enumerate() {
        match c {
            '\\' => inside_escape = true,
            '[' => { inside_brackets = true && !inside_escape; inside_escape = false; if !buffer.is_empty(){ if let Some(start) = start_pos { add_litteral(&mut literals, buffer.clone(), start);} buffer.clear();}}, // Ignore tout ce qui est dans les crochets []
            ']' => { inside_brackets = false; inside_escape= false; if !buffer.is_empty(){ if let Some(start) = start_pos { add_litteral(&mut literals, buffer.clone(), start);} buffer.clear();}}, // Fin de la zone entre []
            '(' => { 
                inside_escape= false;
                if !buffer.is_empty() {
                    if let Some(start) = start_pos {
                        add_litteral(&mut literals, buffer.clone(), start);
                    }
                    buffer.clear();
                }
                },
            ')' => {
                    inside_escape= false;
                    if !buffer.is_empty() {
                        if let Some(start) = start_pos {
                            add_litteral(&mut literals, buffer.clone(), start);
                        }
                        buffer.clear();
                    }
                }
            '{' => {
                    inside_parenthesis = true && !inside_escape;
                    inside_escape= false;
                    if !buffer.is_empty() {
                        if let Some(start) = start_pos {
                            add_litteral(&mut literals, buffer.clone(), start);
                        }
                        buffer.clear();
                    }
                   
                }
            '}' => {
                inside_escape= false;
                inside_parenthesis = false;
                if !buffer.is_empty(){ if let Some(start) = start_pos { add_litteral(&mut literals, buffer.clone(), start);} buffer.clear();}
            }
            '"' | ',' | '-' | '_' | '.' | '*' | '+' | '|' => {inside_escape = false; if !buffer.is_empty(){ if let Some(start) = start_pos { add_litteral(&mut literals, buffer.clone(), start);} buffer.clear();};}, // Exclure les symboles comme guillemets, parenthèses etc.
            _ if inside_brackets => continue, // Ne rien faire à l'intérieur des crochets
            _ if inside_parenthesis => continue,

            _ if c.is_alphanumeric()   => {// Littéraux valides : alphanumérique,
                if count_escape > 0 { count_escape -= 1; continue;}
                if !inside_escape {
                    if buffer.is_empty() {
                        start_pos = Some(i);
                    }
                    buffer.push(c);
                }
                if inside_escape {
                    if c == 'x' {
                        count_escape = 2;
                    } else if c == 'u' {
                        count_escape = 4;
                    }
                }
                
                inside_escape = false;
               
                
            }
            _ if c == '?' && !inside_escape => { // Si on rencontre un '?', le précédent est optionnel, donc on marque cette situation.
                if !buffer.is_empty(){
                    let last = buffer.pop().unwrap();
                    if let Some(start) = start_pos {
                        add_litteral(&mut literals, buffer.clone(), start);
                        start_pos = Some(start + buffer.len());
                        
                    }
                    if let Some(start) = start_pos {
                        
                        add_litteral(&mut literals, last.to_string(), start);
                    }

                    buffer.clear();
                }
            }
            _ => {
                if !buffer.is_empty() {
                    if let Some(start) = start_pos {
                        add_litteral(&mut literals, buffer.clone(), start);
                    }
                    buffer.clear();
                    
                }
            }
        }
    }

    // Si un littéral reste à la fin
    if !buffer.is_empty() {
        if let Some(start) = start_pos {
            add_litteral(&mut literals, buffer.clone(), start);
        }
    }
   
    literals
}

pub fn replace_literals(regex: &str, replacements: &[(String, String, Vec<usize>)]) -> String {
    let mut modified = String::with_capacity(regex.len());
    let mut last_index = 0;

    // Étape 1: Aplatir les remplacements en (position, &original, &replacement)
    let mut flat_replacements: Vec<_> = replacements.iter()
        .flat_map(|(literal, update, positions)| 
            positions.iter().map(move |&p| (p, literal.as_str(), update.as_str()))
        )
        .collect();

    // Étape 2: Trier par position croissante
    flat_replacements.sort_by_key(|&(p, _, _)| p);

    let regex_bytes = regex.as_bytes();

    // Étape 3: Reconstruire la chaîne avec les remplacements
    for (pos, original, replacement) in flat_replacements {
        if pos >= last_index {
            // Ajouter la partie avant le remplacement
            modified.push_str(std::str::from_utf8(&regex_bytes[last_index..pos]).unwrap_or(""));
            // Ajouter le remplacement
            modified.push_str(std::str::from_utf8(replacement.as_bytes()).unwrap_or(""));
            // Mettre à jour last_index
            last_index = pos + original.len();
        }
    }

    // Ajouter la fin restante
    modified.push_str(&regex[last_index..]);

    modified
}


#[cfg(test)]
mod test {
    use super::*;
    
    #[test]
    fn test_extract_litterals(){

        let regexes: Vec<(&str, Vec<&str>)> = vec![
            ("file-name", vec!["file", "name"]),
            (r#"\dhttps?"#, vec!["http", "s"]),
            (r#"aze-zdz\d{1,5}"#, vec!["aze","zdz"]),
            (r#"\{[ ]?"name"[ ]?:[ ]?"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*"[ ]?,[ ]?"age"[ ]?:[ ]?(-)?(0|[1-9][0-9]*)[ ]?,[ ]?"complexe_phone"[ ]?:[ ]?("\+?\d{1,4}?[-. ]?\(\d{1,3}\)?[-. ]?\d{1,4}[-. ]?\d{1,4}[-. ]?\d{1,9}")[ ]?\}"#, 
            vec!["name", "age", "0", "complexe", "phone"]),
            (r#"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"#, vec!["25", "2", "25", "2"]),
            (r###"\{[ ]?"id"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[ ]?,[ ]?"work"[ ]?:[ ]?\{([ ]?"id"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?|([ ]?"id"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[ ]?,)?[ ]?"name"[ ]?:[ ]?"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*"|([ ]?"id"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[ ]?,)?([ ]?"name"[ ]?:[ ]?"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*"[ ]?,)?[ ]?"composer"[ ]?:[ ]?\{[ ]?"id"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[ ]?,[ ]?"name"[ ]?:[ ]?"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*"[ ]?,[ ]?"functions"[ ]?:[ ]?\[[ ]?(("([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*")(,[ ]?("([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*")){0,})?[ ]?\][ ]?\})?[ ]?\}[ ]?,[ ]?"recording_artists"[ ]?:[ ]?\[[ ]?((\{[ ]?"id"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[ ]?,[ ]?"name"[ ]?:[ ]?"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*"[ ]?,[ ]?"functions"[ ]?:[ ]?\[[ ]?(("([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*")(,[ ]?("([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*")){0,})?[ ]?\][ ]?\})(,[ ]?(\{[ ]?"id"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[ ]?,[ ]?"name"[ ]?:[ ]?"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*"[ ]?,[ ]?"functions"[ ]?:[ ]?\[[ ]?(("([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*")(,[ ]?("([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*")){0,})?[ ]?\][ ]?\})){0,})?[ ]?\][ ]?\}"###, 
            vec!["id", "0", "work", "id", "0", "id", "0", "name", "id", "0", "name", "composer", "id", "0", "name", "functions", "recording", "artists", "id", "0", "name", "functions", "id", "0", "name", "functions"]),
            (r#""[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}""#, vec![]),
            (r#"(true|false)"#, vec!["true", "false"])
        
        ];

        for (regex, expected) in regexes {
            let literals = extract_literals(regex);
            let extracted: Vec<String> = literals
                .into_iter()
                .map(|(lit, _pos)| lit)  // On récupère juste les littéraux sans la position
                .collect();
            
            assert_eq!(extracted, expected, "Failed for regex: {} - extract: {:?} - expected : {:?}", regex, extracted, expected);
        }
    
   
       
        //println!("Literals extracted: {:?}", literals);


    //     let replacements: Vec<(String, usize, String)> = literals
    //     .iter()
    //     .enumerate()
    //     .map(|(i, (literal, pos))| (literal.clone(), *pos, format!("(\x1C{})", "1".repeat(i as usize))))
    //     .collect();

    
    //     let modified_pattern = replace_literals(regex_pattern, &replacements);
    //     println!("Modified regex: \"{}\"", modified_pattern);
    }

    
}