import timeit
from outlines_core import Index, Vocabulary

regex_samples = {
    "email": r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
    # Ajoute d'autres regex si nécessaire
}

# Initialisation du Vocabulary avant la mesure
vocabulary = Vocabulary.from_pretrained("unsloth/Llama-3.1-8B-Instruct")
pattern = regex_samples["email"]
# Code de setup (ne contient que l'importation et la définition de pattern)
setup_code = "from outlines_core import Index"
# Mesure uniquement la construction de l'Index
stmt = "Index(pattern, vocabulary)"
execution_time = timeit.timeit(stmt, setup=setup_code, globals=locals(), number=1)
print(f"Temps d'exécution pour une construction froide de l'Index (Vocabulary pré-initialisé) : {execution_time} secondes")
