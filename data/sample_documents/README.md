# 示例文档

将你的领域文档放在这里,系统会通过 ETL 流程把它们处理后写入向量库。

## 支持格式

- `.pdf` - 用 `pypdf` 或 `pdfplumber` 解析
- `.docx` - 用 `python-docx` 解析
- `.md` / `.txt` - 直接读取
- `.html` - 用 `beautifulsoup4` 解析

## 使用方法

```python
from ragagent.ingest import DocumentIngestor

ingestor = DocumentIngestor(
    chunk_size=500,
    chunk_overlap=50,
    user_id="default",
)
ingestor.ingest_directory("data/sample_documents/")
```
