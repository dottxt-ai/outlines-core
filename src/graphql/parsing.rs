use apollo_compiler::ast::Type;
use apollo_compiler::schema::EnumType;
use apollo_compiler::schema::ExtendedType;
use apollo_compiler::schema::InterfaceType;
use apollo_compiler::schema::ObjectType;
use apollo_compiler::schema::ScalarType;
use apollo_compiler::schema::UnionType;
use apollo_compiler::Schema;
use regex::escape;

use super::error::GraphQLParserError;
use super::types;

type Result<T> = std::result::Result<T, GraphQLParserError>;

pub(crate) struct Parser<'a> {
    root: Schema,
    whitespace_pattern: &'a str,
    recursion_depth: usize,
    max_recursion_depth: usize,
}

impl<'a> Parser<'a> {
    // Max recursion depth is defined at level 3.
    // Defining recursion depth higher than that should be done cautiously, since
    // each +1 step on the depth blows up regex's size exponentially.
    //
    // For example, for simple referential json schema level 5 will produce regex size over 700K,
    // which seems counterproductive and likely to introduce performance issues.
    // It also breaks even `regex` sensible defaults with `CompiledTooBig` error.
    pub fn new(root: &'a str) -> Result<Self> {
        let root = Schema::parse_and_validate(root, "sdl.graphql")
            .map_err(|err| GraphQLParserError::ApolloCompiler(err.to_string()))?
            .into_inner();

        if root.schema_definition.query.is_none() {
            return Err(GraphQLParserError::UnknownQuery);
        }

        Ok(Self {
            root,
            whitespace_pattern: types::WHITESPACE,
            recursion_depth: 0,
            max_recursion_depth: 3,
        })
    }

    pub fn with_whitespace_pattern(self, whitespace_pattern: &'a str) -> Self {
        Self {
            whitespace_pattern,
            ..self
        }
    }

    #[allow(dead_code)]
    pub fn with_max_recursion_depth(self, max_recursion_depth: usize) -> Self {
        Self {
            max_recursion_depth,
            ..self
        }
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_regex(&mut self) -> Result<String> {
        let query_obj_name = self
            .root
            .schema_definition
            .query
            .as_ref()
            .ok_or_else(|| GraphQLParserError::UnknownQuery)?;

        let query_type = self
            .root
            .types
            .get(&query_obj_name.name)
            .ok_or_else(|| GraphQLParserError::UnknownQuery)?;

        self.parse_type(query_type)
    }

    fn parse_type(&self, node: &ExtendedType) -> Result<String> {
        match node {
            ExtendedType::Scalar(node) => self.parse_scalar(node),
            ExtendedType::Object(node) => self.parse_object(node),
            ExtendedType::Interface(node) => self.parse_interface(node),
            ExtendedType::Union(node) => self.parse_union(node),
            ExtendedType::Enum(node) => self.parse_enum(node),
            ExtendedType::InputObject(_node) => {
                Err(GraphQLParserError::InputValueDefinitionNotSupported)
            }
        }
    }

    fn parse_scalar(&self, node: &ScalarType) -> Result<String> {
        match node.name.as_str() {
            "Int" => Ok(types::INTEGER.to_string()),
            "Float" => Ok(types::NUMBER.to_string()),
            "String" => Ok(types::STRING.to_string()),
            "Boolean" => Ok(types::BOOLEAN.to_string()),
            "ID" => Ok(types::STRING.to_string()),
            "Date" => Ok(types::DATE.to_string()),
            "Uri" => Ok(types::URI.to_string()),
            "Uuid" => Ok(types::UUID.to_string()),
            "Email" => Ok(types::EMAIL.to_string()),
            _ => Ok(types::STRING.to_string()),
        }
    }

    fn parse_object(&self, node: &ObjectType) -> Result<String> {
        let mut regex = String::from(r"\{");

        let last_required_pos = node
            .fields
            .iter()
            .enumerate()
            .filter_map(|(i, (_field_name, field_def))| {
                if matches!(field_def.ty, Type::NonNullList(_) | Type::NonNullNamed(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .max();

        match last_required_pos {
            // We have required fields
            Some(last_required_pos) => {
                for (i, (field_name, field_def)) in node.fields.iter().enumerate() {
                    let mut subregex = format!(
                        r#"{0}"{1}"{0}:{0}"#,
                        self.whitespace_pattern,
                        escape(field_name.as_str())
                    );

                    // TODO: add * for list
                    let (inner_ty_regex, is_required) = self.parse_inner_type(&field_def.ty)?;
                    subregex += &inner_ty_regex;

                    if i < last_required_pos {
                        subregex = format!("{}{},", subregex, self.whitespace_pattern)
                    } else if i > last_required_pos {
                        subregex = format!("{},{}", self.whitespace_pattern, subregex)
                    }

                    if is_required {
                        regex += &subregex;
                    } else {
                        regex += &format!("({})?", subregex);
                    };
                }
            }
            // We don't have any required fields
            None => {
                let mut property_subregexes = Vec::with_capacity(node.fields.len());

                for (field_name, field_def) in &node.fields {
                    let mut subregex = format!(
                        r#"{0}"{1}"{0}:{0}"#,
                        self.whitespace_pattern,
                        escape(field_name.as_str())
                    );

                    let (inner_ty_regex, _is_required) = self.parse_inner_type(&field_def.ty)?;
                    subregex += &inner_ty_regex;

                    property_subregexes.push(subregex);
                }

                let mut possible_patterns = Vec::new();
                for i in 0..property_subregexes.len() {
                    let mut pattern = String::new();
                    for subregex in &property_subregexes[..i] {
                        pattern += &format!("({}{},)?", subregex, self.whitespace_pattern);
                    }
                    pattern += &property_subregexes[i];
                    possible_patterns.push(pattern);
                }

                regex += &format!("({})?", possible_patterns.join("|"));
            }
        }
        regex += &format!("{}\\}}", self.whitespace_pattern);

        Ok(regex)
    }

    fn parse_inner_type(&self, ty: &Type) -> Result<(String, bool)> {
        let mut subregex = String::new();
        let mut is_required = false;
        match ty {
            Type::Named(name) => {
                let ty = self
                    .root
                    .types
                    .get(name)
                    .ok_or_else(|| GraphQLParserError::UnknownType(name.clone()))?;
                subregex += &format!("({})?", self.parse_type(ty)?);
            }
            Type::List(ty) => {
                subregex += &format!(
                    r"\[{0}(({1})(,{0}({1})){{0,}}){0}\]?",
                    self.whitespace_pattern,
                    self.parse_inner_type(ty)?.0
                );
            }
            Type::NonNullNamed(name) => {
                is_required = true;
                let ty = self
                    .root
                    .types
                    .get(name)
                    .ok_or_else(|| GraphQLParserError::UnknownType(name.clone()))?;
                subregex += &self.parse_type(ty)?;
            }
            Type::NonNullList(ty) => {
                is_required = true;
                subregex += &format!(
                    r"\[{0}(({1})(,{0}({1})){{0,}}){0}\]",
                    self.whitespace_pattern,
                    self.parse_inner_type(ty)?.0
                );
            }
        }

        Ok((subregex, is_required))
    }

    fn parse_interface(&self, node: &InterfaceType) -> Result<String> {
        todo!()
    }

    fn parse_union(&self, node: &UnionType) -> Result<String> {
        todo!()
    }

    fn parse_enum(&self, node: &EnumType) -> Result<String> {
        let variants = node
            .values
            .iter()
            .map(|(_name, def)| format!(r#""{}""#, def.value.as_str()))
            .collect::<Vec<String>>()
            .join("|");

        Ok(format!(r"({variants})"))
    }
}
