# Network Whitelist For ProtSent And Similar Protein/DNA LM Workflows

This document is intended for IT / cluster / firewall teams.

Recommendation: whitelist each domain below together with all subdomains when
policy allows. Most traffic is HTTPS on port 443. Large model and dataset
downloads often redirect through vendor CDNs, object stores, or content-addressed
backends under the same parent domain.

## Short Version

If you need a compact allowlist first, start with these:

This shortlist is domain-only. Models and datasets are not separate domains;
they mostly ride over Hugging Face, GitHub, EBI, STRING, UniProt, RCSB, NCBI,
and related hosts.

- `*.huggingface.co`
- `*.hf.co`
- `*.xethub.hf.co`
- `ftp.ebi.ac.uk`
- `afdb-cluster.steineggerlab.workers.dev`
- `stringdb-downloads.org`
- `rest.uniprot.org`
- `www.uniprot.org`
- `*.rcsb.org`
- `files.rcsb.org`
- `*.ncbi.nlm.nih.gov`
- `ftp.ncbi.nlm.nih.gov`
- `*.ensembl.org`
- `ftp.ensembl.org`
- `genome.ucsc.edu`
- `hgdownload.soe.ucsc.edu`
- `pypi.org`
- `files.pythonhosted.org`
- `github.com`
- `api.github.com`
- `raw.githubusercontent.com`
- `objects.githubusercontent.com`
- `release-assets.githubusercontent.com`
- `marketplace.visualstudio.com`
- `*.gallery.vsassets.io`
- `*.gallerycdn.vsassets.io`
- `update.code.visualstudio.com`
- `*.vo.msecnd.net`
- `api.anthropic.com`
- `claude.ai`
- `console.anthropic.com`
- `api.openai.com`
- `chatgpt.com`
- `platform.openai.com`
- `openai.com`
- `cdn.oaistatic.com`
- `conda.anaconda.org`
- `repo.anaconda.com`
- `download.pytorch.org`
- `developer.download.nvidia.com`

If HF namespace-level approval is required, start with these models and datasets:

- Models: `facebook/*`, `Synthyra/*`, `oriel9p/*`, `chandar-lab/*`, `Rostlab/*`, `InstaDeepAI/*`, `zhihan1996/*`, `AIRI-Institute/*`
- Datasets: `OATML-Markslab/*`, `willdaspit/*`, `biomap-research/*`, `proteinea/*`, `AI4Protein/*`, `SaProtHub/*`, `tattabio/*`, `mila-intel/*`, `cradle-bio/*`, `Synthyra/*`

## Required For ProtSent

| Domain or wildcard | Why it is needed | Examples tied to this repo |
| --- | --- | --- |
| `*.huggingface.co` | Hugging Face Hub API, model and dataset metadata, file resolution | ProtSent models, ProteinGym, benchmark datasets, Synthyra models |
| `*.hf.co` | Hugging Face short links and CDN entrypoints | `hf://datasets/willdaspit/afdb_clustered_seqs/**/*.parquet` |
| `*.xethub.hf.co` | Hugging Face large-file CAS / transfer backend for newer large repos | Large models and parquet shards on HF |
| `ftp.ebi.ac.uk` | Pfam training inputs and HMMs | `Pfam-A.fasta.gz`, `Pfam-A.clans.tsv.gz`, `Pfam-A.hmm.gz` |
| `afdb-cluster.steineggerlab.workers.dev` | AFDB Foldseek cluster mapping | `1-AFDBClusters-repId_entryId_cluFlag_taxId.tsv.gz` |
| `stringdb-downloads.org` | STRING sequence and interaction downloads | `protein.sequences.v12.0.fa.gz`, `protein.physical.links.full.v12.0.txt.gz` |
| `pypi.org` | Python package index | `uv sync`, `pip install -e .` |
| `files.pythonhosted.org` | Python wheel and sdist downloads | Packages locked in `uv.lock` |
| `conda.anaconda.org` | MMseqs2 install path if using Bioconda | `conda install -c bioconda mmseqs2` |
| `repo.anaconda.com` | Conda solver packages and base channels | Common Bioconda / conda setup |

## Hugging Face Namespaces Used By This Repo

These are not separate domains, but they are the concrete model and dataset
namespaces currently referenced by ProtSent. If your security team scopes access
inside Hugging Face, this is the starting set.

### Models used or referenced

- `oriel9p/protsent-esm2-35M`
- `oriel9p/protsent-esm2-150M`
- `facebook/esm2_t6_8M_UR50D`
- `facebook/esm2_t12_35M_UR50D`
- `facebook/esm2_t30_150M_UR50D`
- `facebook/esm2_t33_650M_UR50D`
- `Synthyra/ESMplusplus_small`
- `Synthyra/ESMplusplus_large`
- `Synthyra/ESM2-8M`
- `Synthyra/ESM2-150M`
- `Synthyra/DPLM2-150M`
- `Synthyra/DPLM2-650M`
- `Synthyra/Profluent-E1-150M`
- `Synthyra/Profluent-E1-300M`
- `Synthyra/Profluent-E1-600M`
- `chandar-lab/AMPLIFY_120M`
- `RaphaelMourad/Mistral-Prot-v1-417M`

### Datasets used or referenced

- `willdaspit/afdb_clustered_seqs`
- `OATML-Markslab/ProteinGym_v1`
- `Synthyra/bernett_gold_ppi`
- `proteinea/solubility`
- `proteinea/deeploc`
- `biomap-research/peptide_HLA_MHC_affinity`
- `biomap-research/metal_ion_binding`
- `biomap-research/material_production`
- `biomap-research/fold_prediction`
- `biomap-research/antibiotic_resistance`
- `biomap-research/temperature_stability`
- `biomap-research/fitness_prediction`
- `biomap-research/stability_prediction`
- `biomap-research/optimal_ph`
- `biomap-research/enzyme_catalytic_efficiency`
- `biomap-research/cloning_clf`
- `mila-intel/ProtST-BinaryLocalization`
- `AI4Protein/EC`
- `AI4Protein/GO_MF`
- `andrewdalpino/CAFA5`
- `cradle-bio/tape-fluorescence`
- `SaProtHub/Dataset-Thermostability-FLIP`
- `SaProtHub/Dataset-Beta_Lactamase-PEER`
- `SaProtHub/Dataset-AAV-FLIP`
- `SaProtHub/DATASET-CAPE-RhlA-seqlabel`
- `tattabio/scope40_test`
- `anonymous-protsent/SignalP_Binary`
- `anonymous-protsent/ProFET_NP_SP_Cleaved`

## Recommended Science Data Domains For Similar Projects

These are not all required for ProtSent today, but they are common for adjacent
protein language model, DNA language model, structure, retrieval, and benchmark
workflows.

| Domain or wildcard | Typical use |
| --- | --- |
| `*.ebi.ac.uk` | Pfam, InterPro, AlphaFold EBI, ENA, MGnify, many EMBL-EBI mirrors |
| `alphafold.ebi.ac.uk` | AlphaFold Protein Structure Database |
| `rest.uniprot.org` | UniProt REST API |
| `www.uniprot.org` | UniProt web and downloads |
| `*.rcsb.org` | PDB / RCSB structure and metadata access |
| `files.rcsb.org` | Direct PDB/mmCIF downloads |
| `data.rcsb.org` | Structured RCSB data API |
| `*.ncbi.nlm.nih.gov` | NCBI sequence, GEO, SRA, PubMed and related APIs |
| `ftp.ncbi.nlm.nih.gov` | Large NCBI reference and annotation downloads |
| `*.ensembl.org` | Ensembl APIs and reference annotations |
| `ftp.ensembl.org` | Ensembl bulk downloads |
| `genome.ucsc.edu` | UCSC Genome Browser |
| `hgdownload.soe.ucsc.edu` | UCSC reference genome and annotation downloads |
| `string-db.org` | STRING web UI and docs |
| `stringdb-downloads.org` | STRING bulk download endpoint |
| `cathdb.info` | CATH structure classification data |
| `scop.berkeley.edu` | SCOP / SCOPe structure classification references |
| `doi.org` | DOI resolution for papers and datasets |
| `arxiv.org` | Paper references and model / benchmark documentation |

## Common Hugging Face Namespaces For Adjacent Protein And DNA LM Work

Again, these are not separate domains. They are useful examples for security
teams that want to pre-approve the most likely model and dataset namespaces.

### Protein-language-model-heavy namespaces

- `facebook/*` for ESM / ESM2
- `Synthyra/*` for ESMplusplus, FastPLM ESM2, DPLM2, Profluent-E1 and benchmark data
- `Rostlab/*` for ProtBERT / ProtT5 family models
- `chandar-lab/*` for AMPLIFY
- `oriel9p/*` for ProtSent outputs

### DNA / nucleotide-heavy namespaces often seen in similar projects

- `InstaDeepAI/*` for Nucleotide Transformer family models
- `zhihan1996/*` for DNABERT2-family checkpoints
- `AIRI-Institute/*` for GENA-LM style genomic models

## AI Coding Assistants And VS Code Extensions

If the same machines are also used for coding agents, extension install/update,
or prompt-driven dev workflows, whitelist these too.

| Domain or wildcard | Typical use |
| --- | --- |
| `marketplace.visualstudio.com` | VS Code extension discovery |
| `*.gallery.vsassets.io` | VS Code extension package retrieval |
| `*.gallerycdn.vsassets.io` | VS Code extension CDN |
| `update.code.visualstudio.com` | VS Code binary / extension update paths |
| `*.vo.msecnd.net` | Microsoft CDN used by VS Code assets |
| `github.com` | Repo cloning, releases, issue links |
| `api.github.com` | GitHub API |
| `raw.githubusercontent.com` | Raw files and installation scripts |
| `objects.githubusercontent.com` | GitHub object / asset delivery |
| `release-assets.githubusercontent.com` | GitHub release asset downloads |
| `*.githubusercontent.com` | GitHub-hosted content and CDN endpoints |
| `api.anthropic.com` | Claude API |
| `claude.ai` | Claude web app |
| `console.anthropic.com` | Anthropic console / key management |
| `api.openai.com` | OpenAI API / Codex-style extension backends |
| `chatgpt.com` | ChatGPT web app |
| `platform.openai.com` | OpenAI console / key management |
| `openai.com` | OpenAI docs and redirects |
| `cdn.oaistatic.com` | OpenAI static assets |
| `open-vsx.org` | Optional extension marketplace for non-Microsoft VS Code builds |

## Packaging, Build, And GPU Tooling

These are useful on build nodes, fresh dev machines, and training boxes.

| Domain or wildcard | Typical use |
| --- | --- |
| `pypi.org` | Python package index |
| `files.pythonhosted.org` | Python wheels and sdists |
| `github.com` | Source checkout and release links |
| `api.github.com` | GitHub API calls from tooling |
| `raw.githubusercontent.com` | Setup scripts and raw file fetches |
| `conda.anaconda.org` | Bioconda / conda-forge channels |
| `repo.anaconda.com` | Main conda channels |
| `download.pytorch.org` | PyTorch wheels and auxiliary indexes |
| `developer.download.nvidia.com` | CUDA, NCCL, cuDNN and NVIDIA tooling downloads |

## Verification Notes

This whitelist was derived from the current ProtSent codebase and docs.

Verified directly from repo references:

- `data_prep.py` downloads from `ftp.ebi.ac.uk`, `afdb-cluster.steineggerlab.workers.dev`, and `stringdb-downloads.org`.
- `data_prep.py` also uses Hugging Face datasets for `willdaspit/afdb_clustered_seqs`, `Synthyra/bernett_gold_ppi`, and `OATML-Markslab/ProteinGym_v1`.
- `benchmark_tasks.py` references the full benchmark dataset set listed above, almost all under Hugging Face.
- `README.md`, `protein_pipeline.py`, `protein_benchmark_suite.py`, `model_utils.py`, and `static_guide.py` reference the model namespaces listed above.
- `uv.lock` and the install flow imply `pypi.org` and `files.pythonhosted.org`.

Operational note from this machine: recent attempts to reach Hugging Face, EBI,
and STRING endpoints timed out. That makes a proper whitelist especially likely
to be part of the root cause.