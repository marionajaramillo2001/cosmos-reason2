# Planning Prompts

This project uses three prompt templates for AgiBot long-horizon planning:

- `planning_direct.yaml`: generate an ordered action plan directly from the initial video state and goal.
- `planning_hierarchical.yaml`: decompose the task into subgoals and executable robot actions.
- `planning_rag.yaml`: generate a plan with retrieved reference examples.

`planning_rag.yaml` is shared by both RAG conditions:

```text
intra_task_rag: retrieved examples come from the same task cluster.
inter_task_rag: retrieved examples come from other task clusters only.
```
