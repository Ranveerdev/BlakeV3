FUNCTION_WORDS = [
    # Determiners / Articles
    "a", "an", "the",
    "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "each", "every", "either", "neither", "some", "any", "no",
    "many", "few", "several", "both", "all", "much", "most",

    # Pronouns
    "i", "me", "you", "he", "him", "she", "her", "it",
    "we", "us", "they", "them",
    "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves",
    "who", "whom", "whose", "which", "what",
    "whoever", "whomever", "whichever", "whatever",
    "someone", "anyone", "no‑one", "everyone", "something",
    "anything", "nothing", "everything", "somebody",
    "anybody", "nobody", "everybody",

    # Auxiliary / Modal verbs
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did",
    "will", "would",
    "can", "could",
    "shall", "should",
    "may", "might",
    "must", "ought",

    # Prepositions
    "about", "above", "across", "after", "against", "along",
    "amid", "among", "around", "at", "before", "behind",
    "below", "beneath", "beside", "between", "beyond",
    "but", "by", "concerning", "despite", "down", "during",
    "except", "for", "from", "in", "inside", "into", "like",
    "near", "of", "off", "on", "onto", "out", "outside",
    "over", "past", "regarding", "since", "through",
    "throughout", "to", "toward", "under", "underneath",
    "until", "up", "upon", "with", "within", "without",

    # Coordinating conjunctions
    "and", "but", "or", "nor", "for", "yet", "so",

    # Subordinating conjunctions
    "after", "although", "as", "because", "before",
    "even if", "even though", "if", "in case", "once",
    "provided", "since", "so that", "than", "that",
    "though", "unless", "until", "when", "whenever",
    "where", "wherever", "whether", "while",

    # Sentence adverbs / discourse markers
    "however", "therefore", "moreover", "meanwhile",
    "nevertheless", "furthermore", "instead", "otherwise",
    "thus", "hence",

    # Interjections / fillers (some linguists treat these as function words)
    "oh", "ah", "uh", "um", "er", "hey", "well"
]

def remove_function_word(array):
    # Use a list comprehension to filter out function words
    return [i for i in array if i not in FUNCTION_WORDS]