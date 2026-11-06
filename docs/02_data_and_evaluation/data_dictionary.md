# Từ Điển Dữ Liệu (Data Dictionary)

## 1. Các thực thể chính (Entities)
- **Compound**: Dapagliflozin (DB06292, PubChem CID 9887712).
- **Gene**: SLC5A2 / SGLT2 (NCBI Gene ID: 6524).
- **Diseases**:
  - T2D: DOID:9352 (Baseline - loại trừ).
  - Heart Failure (HFrEF): DOID:9333 / MeSH D006333 (Hard Positive 1).
  - Chronic Kidney Disease (CKD): DOID:784 / MeSH D051436 (Hard Positive 2).
  - NAFLD / NASH: DOID:0080358 (Soft Positive).

## 2. 5 Hợp đồng dữ liệu (Data Contracts A -> E)
- **Contract A**: DocumentRecord (Huy -> Khang)
- **Contract B**: ExtractedRelation (Khang -> Huy)
- **Contract C**: NormalizedRelation (Huy + Khang -> KG)
- **Contract D**: GraphRecord & Evidence (Huy -> Liam)
- **Contract E**: Candidate & Explanation (Liam -> UI)
