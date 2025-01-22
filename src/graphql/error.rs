use apollo_compiler::validation::DiagnosticList;
use apollo_compiler::Name;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GraphQLParserError {
    #[error("GraphQL apollo compiler error: {0}")]
    ApolloCompiler(String),
    #[error("Can't find any Query type in your schema, it must exists and is the entrypoint")]
    UnknownQuery,
    #[error("Can't find any type {0} in your schema")]
    UnknownType(Name),
    #[error("Input value definition is not supported")]
    InputValueDefinitionNotSupported,
}
