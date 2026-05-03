"""Utilities for gene annotation and identifier mapping."""

from typing import Iterable
import warnings

from loguru import logger
import mygene
import pandas as pd


# ----------------------------------------------------------------------
# 1. Low-level alias retrieval
# ----------------------------------------------------------------------
def fetch_gene_aliases(
    genes: Iterable[str],
    species: str = "human",
) -> pd.Series:
    """Fetch aliases for HUGO/HGNC symbols from MyGene.info.

    Parameters
    ----------
    genes : Iterable[str]
        Official HUGO/HGNC gene symbols.
    species : str, default="human"
        Species name passed to MyGene.

    Returns
    -------
    pandas.Series
        Series indexed by HUGO symbol whose values are ``list[str]`` of aliases.
        Empty Series is returned when MyGene yields no ``alias`` field.

    Notes
    -----
    MyGene occasionally returns duplicated index rows for the same symbol;
    only the first occurrence is kept.
    """
    mg = mygene.MyGeneInfo()
    df = mg.querymany(
        list(genes),
        species=species,
        scopes=["symbol"],
        fields=["alias", "symbol"],
        as_dataframe=True,
        df_index=True,
        verbose=False,
    )
    if "alias" not in df.columns or df.empty:
        return pd.Series(dtype=object)

    df = df[~df.index.duplicated(keep="first")]
    return df["alias"].dropna()


# ----------------------------------------------------------------------
# 2. HUGO -> target mapper
# ----------------------------------------------------------------------
def build_hugo_to_target_mapper(
    hugo_genes: Iterable[str],
    target_genes: Iterable[str],
) -> tuple[dict[str, str], set[str], set[str]]:
    """Build a mapping from HUGO symbols to names from ``target_genes``.

    The resolution proceeds in two passes:

    1. Direct intersection ``hugo_genes ∩ target_genes``.
    2. For the remainder, MyGene aliases are queried and intersected with
       the unused portion of ``target_genes``.

    A 1-to-1 invariant is enforced: each target name is consumed by at most
    one HUGO symbol to avoid silent collisions.

    Parameters
    ----------
    hugo_genes : Iterable[str]
        Source HUGO/HGNC symbols (e.g. ``df.columns`` or ``adata.var_names``).
    target_genes : Iterable[str]
        Target naming pool (alias / legacy / platform-specific identifiers).

    Returns
    -------
    mapper : dict[str, str]
        Mapping ``hugo_symbol -> target_name``. For unresolved symbols the
        value falls back to the original HUGO name (identity).
    found : set[str]
        HUGO symbols successfully resolved (directly or via alias).
    missing : set[str]
        HUGO symbols with no match in ``target_genes``.

    Warns
    -----
    UserWarning
        Raised when more than one alias of a single HUGO symbol matches the
        target pool; the lexicographically first hit is selected.
    """
    hugo_set = set(hugo_genes)
    target_set = set(target_genes)

    direct = hugo_set & target_set
    mapper: dict[str, str] = {g: g for g in direct}
    found: set[str] = set(direct)
    missing: set[str] = set()

    remaining_hugo = hugo_set - target_set
    remaining_target = target_set - hugo_set  # consumed pool

    logger.debug(
        f"Direct: {len(direct)} | resolving {len(remaining_hugo)} via MyGene aliases"
    )

    if not remaining_hugo:
        return mapper, found, missing

    aliases = fetch_gene_aliases(remaining_hugo)

    for gene in remaining_hugo:
        if gene not in aliases.index:
            missing.add(gene)
            mapper[gene] = gene
            continue

        raw = aliases.loc[gene]
        # MyGene may return str, list, or Series (on duplicated indices).
        if isinstance(raw, list):
            alias_set = set(raw)
        elif isinstance(raw, pd.Series):
            alias_set = set(raw.dropna().tolist())
        else:
            alias_set = {raw}

        hits = alias_set & remaining_target

        if len(hits) == 1:
            tgt = next(iter(hits))
        elif len(hits) > 1:
            warnings.warn(f"{len(hits)} alias hits for {gene}: {hits}; picking first")
            tgt = sorted(hits)[0]
        else:
            missing.add(gene)
            mapper[gene] = gene
            continue

        mapper[gene] = tgt
        found.add(gene)
        remaining_target.discard(tgt)

    logger.info(
        f"Resolved via alias: {len(found) - len(direct)} | missing: {len(missing)}"
    )

    return mapper, found, missing


# ----------------------------------------------------------------------
# 3. DataFrame column renaming
# ----------------------------------------------------------------------
def rename_df_columns_via_aliases(
    df: pd.DataFrame,
    target_genes: Iterable[str],
    drop_missing: bool = False,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    """Rename DataFrame columns from HUGO format to ``target_genes`` format.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose columns hold HUGO/HGNC gene symbols.
    target_genes : Iterable[str]
        Target naming pool (e.g. ``adata.var_names`` from a different platform).
    drop_missing : bool, default=False
        If True, columns with no resolved match are dropped from the output.
    verbose : bool, default=False
        Print resolution stats.

    Returns
    -------
    renamed_df : pandas.DataFrame
        DataFrame with columns renamed according to the resolved mapper.
    found : set[str]
        HUGO symbols successfully resolved.
    missing : set[str]
        HUGO symbols left unresolved.

    Notes
    -----
    Two distinct HUGO symbols may collapse onto the same target name and
    produce duplicated columns in ``renamed_df``. Validate with
    ``renamed_df.columns.is_unique`` and aggregate (e.g. ``groupby`` on
    ``axis=1``) when needed.
    """
    mapper, found, missing = build_hugo_to_target_mapper(df.columns, target_genes)

    if drop_missing and missing:
        df = df.drop(columns=[c for c in df.columns if c in missing])
        mapper = {k: v for k, v in mapper.items() if k not in missing}

    return df.rename(columns=mapper), found, missing
