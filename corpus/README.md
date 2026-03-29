# Corpus

Standaardmanifest bevat alleen **PDF’s** van **Open Overheid** (Defensie DS/AI-strategie + overheidsbrede handreiking generatieve AI). Ingest gebruikt **PyMuPDF** per pagina.

## Bestanden ophalen

PDF’s staan onder **`corpus/raw/`** (gitignored). Na een clone:

```bash
./scripts/download_corpus.sh
rag-ingest --reset
```

## Meer documenten

Zet PDF’s in `corpus/raw/`, voeg entries toe in `corpus/manifest.yaml`, en draai opnieuw `rag-ingest --reset`. Ondersteunde formaten in code: **`.pdf`**, **`.md`**, **`.txt`** (geen HTML).

### EUR-Lex / EU-wettekst

Geautomatiseerde downloads van `eur-lex.europa.eu` falen vaak (WAF). Download het PDF in de browser en plaats het handmatig in `corpus/raw/`, bv. [AI Act (CELEX:32024R1689)](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689).

## Demo-markdown

Optioneel: `corpus/sample_documents/policy_demo_nl.md` (fictief) — niet in het standaard-manifest; alleen voor lokale tests.
