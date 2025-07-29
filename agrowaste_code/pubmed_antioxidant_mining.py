

# Load a biomedical NLP model (SciSpacy)
# pip install scispacy https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
# import scispacy
# nlp = spacy.load("en_core_sci_sm")

# For demonstration, we'll use spaCy's small English model
nlp = spacy.load("en_core_web_sm")

# Configure Entrez
Entrez.email = os.getenv("ENTREZ_EMAIL", "your.email@example.com")


def fetch_pmids(query: str, max_results: int = 200) -> List[str]:
    """
    Search PubMed and return a list of PMIDs for the given query.
    """
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    return record["IdList"]


def fetch_abstracts(pmids: List[str]) -> Dict[str, str]:
    """
    Fetch abstracts for each PMID and return a dict mapping PMID to abstract text.
    """
    abstracts = {}
    for pmid in pmids:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        text = handle.read()
        abstracts[pmid] = text
    return abstracts


def extract_metrics(text: str) -> Dict[str, Optional[str]]:
    """
    Parse the abstract text to find IC50 (DPPH), TPC values, sample names, and proximate composition data.

    Returns:
        {
            'ic50_dpph_value': '...',
            'ic50_dpph_unit': '...',
            'tpc_value': '...',
            'tpc_unit': '...',
            'sample_name': '...',
            'proximal_composition': '...'  # full text or parsed list
        }
    """
    results = {
        'ic50_dpph_value': None,
        'ic50_dpph_unit': None,
        'tpc_value': None,
        'tpc_unit': None,
        'sample_name': None,
        'proximal_composition': None
    }

    # Simplified regex patterns
    ic50_pattern = re.search(r"IC50\s*(?:by DPPH)?\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*(µ?g/mL|mg/mL|µM)", text, re.IGNORECASE)
    if ic50_pattern:
        results['ic50_dpph_value'] = ic50_pattern.group(1)
        results['ic50_dpph_unit'] = ic50_pattern.group(2)

    tpc_pattern = re.search(r"total phenolic (?:content|content\(TPC\))\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*(mg GAE/g|g GAE/100g|µmol GAE/g)", text, re.IGNORECASE)
    if tpc_pattern:
        results['tpc_value'] = tpc_pattern.group(1)
        results['tpc_unit'] = tpc_pattern.group(2)

    # Use NLP to extract sample/residue name: look for phrases like "extract of X"
    doc = nlp(text)
    for sent in doc.sents:
        if 'extract of' in sent.text.lower():
            # crude extraction: take phrase after 'extract of'
            match = re.search(r"extract of ([A-Za-z0-9\s-]+?)[,\.]", sent.text, re.IGNORECASE)
            if match:
                results['sample_name'] = match.group(1).strip()
                break

    # Proximal composition: look for macronutrient percentages
    prox_pattern = re.findall(r"(protein|fat|ash|fiber|carbohydrate)\s*[:=]?\s*([0-9]+\.?[0-9]*)%", text, re.IGNORECASE)
    if prox_pattern:
        # join as semicolon-separated list
        comp_list = [f"{nutrient}:{value}%" for nutrient, value in prox_pattern]
        results['proximal_composition'] = "; ".join(comp_list)

    return results


def build_dataset(query: str, max_papers: int = 200) -> pd.DataFrame:
    """
    Main pipeline: search, fetch abstracts, extract metrics, and assemble a DataFrame.
    """
    pmids = fetch_pmids(query, max_results=max_papers)
    abstracts = fetch_abstracts(pmids)

    records = []
    for pmid, text in abstracts.items():
        metrics = extract_metrics(text)
        metrics['pmid'] = pmid
        records.append(metrics)

    df = pd.DataFrame.from_records(records)
    return df


if __name__ == "__main__":
    # Example usage
    df = build_dataset("agro-industrial waste antioxidant DPPH IC50 TPC", max_papers=100)
    df.to_csv("antioxidant_dataset.csv", index=False)
    print("Dataset saved to antioxidant_dataset.csv")
