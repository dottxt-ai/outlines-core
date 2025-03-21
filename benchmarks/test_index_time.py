import timeit
from outlines_core import Index, Vocabulary, json_schema

regex_samples = {
    "email": r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
    # Ajoute d'autres regex si nécessaire
    "schema_phone": r'''{
                "$schema": "http://json-schema.org/draft-04/schema#",
                "title": "Schema for a Curriculum Vitae",
                "type": "object",
                "definitions": {
                    "experienceEntry": {
                    "type": "object",
                    "properties": {
                        "date": {
                        "type": "string",
                        "format": "date"
                        },
                        "position": {
                        "type": "string"
                        }
                    },
                    "required": ["date", "position"]
                    }
                },
                "properties": {
                    "name": {
                    "type": "string"
                    },
                    "surname": {
                    "type": "string"
                    },
                    "email": {
                    "type": "string",
                    "pattern": "[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
                    },
                    "phone": {
                    "type": "string",
                    "pattern": "\\+?\\d{1,4}?[-. ]?\\(\\d{1,3}\\)?[-. ]?\\d{1,4}[-. ]?\\d{1,4}[-. ]?\\d{1,9}"
                    },
                    "website": {
                    "type": "string",
                    "pattern": "(https?:\\/\\/)?([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([\\/\\w \\.-]*)*\\/?"
                    },
                    "resume": {
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/experienceEntry"
                    }
                    }
                },
                "required": ["name", "surname", "email", "phone", "resume"]
                }'''
}

# Initialisation du Vocabulary avant la mesure
vocabulary = Vocabulary.from_pretrained("unsloth/Llama-3.1-8B-Instruct")
#pattern = regex_samples["email"]
pattern = json_schema.build_regex_from_schema(regex_samples['schema_phone'])
# Code de setup (ne contient que l'importation et la définition de pattern)
setup_code = "from outlines_core import Index"
# Mesure uniquement la construction de l'Index
stmt = "Index(pattern, vocabulary)"
execution_time = timeit.timeit(stmt, setup=setup_code, globals=locals(), number=1)
print(f"Temps d'exécution pour une construction froide de l'Index (Vocabulary pré-initialisé) : {execution_time} secondes")
