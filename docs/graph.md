```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__(<p>__start__</p>)
	analyze_query(analyze_query)
	retrieve(retrieve)
	grade_documents(grade_documents)
	rewrite_query(rewrite_query)
	generate(generate)
	fallback(fallback)
	web_search(web_search)
	hallucination_check(hallucination_check)
	__end__(<p>__end__</p>)
	__start__ --> analyze_query;
	analyze_query --> retrieve;
	grade_documents -.-> fallback;
	grade_documents -.-> generate;
	grade_documents -.-> rewrite_query;
	grade_documents -.-> web_search;
	retrieve --> grade_documents;
	rewrite_query --> retrieve;
	web_search -.-> fallback;
	web_search -.-> generate;
	fallback --> __end__;
	generate --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
