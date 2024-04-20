 # LLM-based multi-agent poetry generation in
non-cooperative environments
(work in progress)
This repository contains the code and data for our paper: [LLM-based multi-agent poetry generation in
non-cooperative environments].

> **Abstract**: 
> Despite the substantial progress of language models, poetry generation remains a difficult task
due to the intricate nature of human expression. Recent advancements in autonomous agents driven
by large language models (LLMs) have proposed the potential to solve complex tasks in a more
human-like way. Under the rationale that the learning process of the poetry generation systems
should be more human-like and divergent to gain creativity, we introduce a framework based
on social learning where we emphasize non-cooperative interactions in addition to cooperative
interaction to encourage divergence. This paper presents the first experiment on an LLM-based
multi-agent system for poetry generation and we utilize both training-based agents (GPT-2) and
conversational agents (GPT-3 and GPT-4). Our evaluation based on 24k generated poems shows
that our framework benefits the poetry generation process for training-based agents resulting in 1)
a 4.7-4.8 percentage points (pp) increase in diversity and a 7-9.5 pp increase in novelty according
to distinct and novel n-grams. The generated poetry from training-based agents also exhibits
group divergence in terms of lexicons, styles and semantics in accordance to the predefined group
affiliation. On the other hand, conversational agents (GPT-3 and GPT-4) in our framework can
generate poetry with very few grammatical errors. Their lexical diversity merely improves at the
first iteration and decreases over time though in comparison conversational agents use more
diverse vocabulary. Moreover, they fail to show any group-based divergence as well as prone to
generating poetry of homogenous styles over time.

## Generated Poetry

We release our [generated poems](dataset/poems). 

## Experiments
To reproduce the evaluations conducted in this work, please check the folder [results for section datasets](results/section_datasets) and [results for section experiments](results/section_experiments).

We provide code for both training-based [training-based](trainable_agents) and conventional frameworks  and [conversational](untrainable_agents). 

## Contacts
If you have any questions, feel free to contact us!

Ran Zhang ([ran.zhang@uni-bielefeld.de](mailto:ran.zhang@uni-mannheim.de))
