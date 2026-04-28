# Expression matrix with HUGO symbols in columns.

# Пример использования

```python
    import pandas as pd
    from batchcor_rna_emb.logging import set_logger
    from batchcor_rna_emb.feature_calc.gene_mapping import rename_df_columns_via_aliases

    expr = pd.DataFrame(
        {
            "TP53":      [1.0, 2.0, 3.0],
            "EGFR":      [0.5, 1.5, 2.5],
            "CCBL1":     [0.1, 0.2, 0.3],   # legacy: new symbol KYAT1
            "HN1L":      [0.4, 0.5, 0.6],   # legacy: new symbol JPT2
            "OBFC1":     [0.7, 0.8, 0.9],   # legacy: new symbol STN1
            "MYUNKNOWN": [0.0, 0.0, 0.0],   # unresolvable
        },
        index=["sample_1", "sample_2", "sample_3"],
    )
 
    # Target pool, e.g. var_names from an AnnData built on a newer Gencode.
    target_names = ["TP53", "EGFR", "KYAT1", "JPT2", "STN1", "BRCA1"]
 
    renamed_df, found, missing = rename_df_columns_via_aliases(
        expr,
        target_genes=target_names,
        drop_missing=False,
    )
 
    logger.info("\nFOUND  :", sorted(found))
    logger.info("MISSING:", sorted(missing))
    logger.info("\nRenamed columns:", list(renamed_df.columns))
    display(renamed_df.head())
```
    
# Potential pitfalls

- **Неоднозначность алиасов. **
Один alias может ссылаться на несколько HUGO (например, легаси CDKN2). Берём первый из отсортированного set + warning — для критичных кейсов лучше валидировать руками.
- **Withdrawn / merged symbols. **
MyGene не всегда отдаёт исторические переименования в alias; для полноты можно дополнять через HGNC dump (prev_symbol поле).
- **Регистр. **
MAPK1 ≠ Mapk1. Если данные из мыши/смешанные — нормализуйте .str.upper() до маппинга.
- **Батчи MyGene. **
Лимит ~1000 на запрос; querymany сам режет, но при списках 50k+ закладывайте время + retries.
- **Дубликаты в df.columns после rename.**
Если два HUGO смаппились в один target — df.rename создаст дублирующиеся колонки. Проверяйте renamed_df.columns.is_unique, при необходимости агрегируйте (groupby(axis=1).mean())