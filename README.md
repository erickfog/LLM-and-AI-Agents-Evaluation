# Seu Modelo Está Bom Mesmo? Como Avaliar LLMs de Verdade

> **Motivação**: Modelos generativos são difíceis de avaliar — você precisa de métricas objetivas, alinhadas ao uso e à arquitetura.

Este repositório apresenta uma abordagem prática para avaliação de Large Language Models (LLMs) e sistemas GenAI, focando em métricas que realmente importam para o seu caso de uso.

## 📋 Índice

- [O Que São Métricas de Avaliação?](#o-que-são-métricas-de-avaliação)
- [Métricas Core vs Métricas Custom](#métricas-core-vs-métricas-custom)
- [LLM-as-a-Judge: Métricas de Nova Geração](#llm-as-a-judge-métricas-de-nova-geração)
- [Exemplos de Métricas Genéricas](#exemplos-de-métricas-genéricas)
- [Métricas Custom e Task-Specific](#métricas-custom-e-task-specific)
- [Avaliação de Agentes](#avaliação-de-agentes)
- [Métricas de RAG](#métricas-de-rag)
- [Como Escolher Métricas](#como-escolher-métricas)
- [Métricas Referenceless (Produção)](#métricas-referenceless-produção)
- [Workflow Prático de Avaliação](#workflow-prático-de-avaliação)
- [Avaliação Humana Complementar](#avaliação-humana-complementar)
- [Métricas de Avaliação de Impacto](#métricas-de-avaliação-de-impacto)
- [Conclusão & Checklist](#conclusão--checklist)
- [Próximos Passos](#próximos-passos)
- [Referências](#referências)

---

## O Que São Métricas de Avaliação?

Métricas de avaliação são ferramentas essenciais para medir o desempenho de um LLM ou sistema GenAI. Elas devem:

- ✔ Medir desempenho de um LLM ou sistema GenAI
- ✔ Ter critérios claros de sucesso
- ✔ Ser indicadores de evolução e regressão nos modelos
- ✔ Poder rodar em pipelines de teste automatizados (CI/CD)

---

## Métricas Core vs Métricas Custom

### Métricas Core (Genéricas)

Métricas genéricas que se aplicam a diversos casos de uso:

- **Answer relevancy** — Relevância da resposta
- **Faithfulness (alucinação)** — Verificação de alucinações
- **Contextual precision/recall** — Precisão e recall contextual
- **Task completion** — Completude da tarefa
- **Tool correctness** — Correção de uso de ferramentas

### Métricas Custom

Métricas personalizadas para casos de uso específicos:

- **GEval** — Critério textual personalizado
- **DAG** — Avaliador baseado em árvore de decisão

> 💡 **Dica prática**: Use até ~5 métricas para não dispersar foco.

---

## LLM-as-a-Judge: Métricas de Nova Geração

Em vez de métricas tradicionais (BLEU/ROUGE), métricas baseadas em **LLM-as-a-Judge** usam outro modelo para julgar qualidade — o que melhora alinhamento com avaliação humana.

### Principais Vantagens

- Mais sensíveis ao significado
- Podem ser customizadas com critérios em linguagem natural
- Menos dependência de referência exata

---

## Exemplos de Métricas Genéricas

### 📍 Answer Relevancy
Quão bem a resposta atende à pergunta.

### 📍 Faithfulness
A resposta está baseada no contexto disponível?

### 📍 Contextual Precision/Recall
Relevância da informação recuperada (RAG).

### 📍 Bias/Toxicity
Avaliação de Segurança e Ética.

---

## Métricas Custom e Task-Specific

### G-Eval
Métrica customizada baseada em cadeia de pensamento.

### DAG
Métricas estruturadas com lógica mais complexa.

### Task-Specific
Métricas de sucesso alinhadas ao caso de uso.

**Exemplo**: Sumarização com critérios de:
- Cobertura
- Precisão
- Não-contradição

---

## Avaliação de Agentes

Para agentes GenAI (que fazem mais do que apenas gerar texto), métricas específicas incluem:

- 🔸 **Task Completion** — Agente completa o objetivo?
- 🔸 **Argument Correctness** — Argumentos para ferramentas corretos?
- 🔸 **Tool Correctness** — Ferramenta correta chamada?
- 🔸 **Plan Quality** — Qualidade do plano de ação
- 🔸 **Plan Adherence** — Seguiu seu próprio plano?
- 🔸 **Step Efficiency** — Eficiência das ações realizadas

---

## Métricas de RAG

Para sistemas de Retrieval-Augmented Generation:

- 📍 **Contextual Precision** — Relevância dos documentos recuperados
- 📍 **Contextual Recall** — Completude da recuperação
- 📍 **Contextual Relevancy** — Relação entre contexto e query
- 📍 **Answer Relevancy** — Resposta final alinhada ao contexto

---

## Como Escolher Métricas

📌 **Não use mais de 5 métricas ao mesmo tempo**

📌 **Misture métricas genéricas e custom**

📌 **Certifique-se de que cada métrica tem um critério claro de sucesso**

---

## Métricas Referenceless (Produção)

Em produção, muitas vezes não existe "o dado de verdade" — então usamos **métricas referenceless**, que apenas observam input/output sem referência anotada.

Podem medir:
- ✔ Consistência
- ✔ Segurança
- ✔ Coerência contextual

---

## Workflow Prático de Avaliação

1. 🔹 **Curar dataset de teste** com "goldens"
2. 🔹 **Escolher métricas apropriadas**
3. 🔹 **Rodar avaliações automatizadas** (DeepEval ou ferramentas equivalentes)
4. 🔹 **Analisar scores e casos falhos**
5. 🔹 **Iterar no modelo/prompt/configuração**

---

## Avaliação Humana Complementar

Mesmo com métricas automáticas, a avaliação humana continua essencial para:

- ✔ Fluência
- ✔ Naturalidade
- ✔ Relevância subjetiva
- ✔ Usabilidade num contexto de aplicação final

---

## Métricas de Avaliação de Impacto

Além de métricas de saída, considere métricas de negócio ou experiência:

- 📌 Satisfação do usuário
- 📌 Redução de tickets
- 📌 Tempo de resolução
- 📌 Aderência ao fluxo de trabalho

---

## Conclusão & Checklist

✅ Métricas alinhadas com objetivo do modelo  
✅ Mistura de métricas genéricas e custom  
✅ Métricas de agente quando aplicável  
✅ Avaliação referenceless para produção  
✅ Combinar avaliação automática e humana

---

## Próximos Passos

### Hands-On com DeepEval

Agora vamos praticar com:
- 📍 DeepEval (ou framework equivalente)
- 📍 Métricas LLM-as-a-Judge
- 📍 Métricas de RAG
- 📍 Métricas de agentes

---

## Referências

- **Confident AI – The Ultimate LLM Evaluation Guide** (blog) — Cobertura de métricas, LLM-as-judge, RAG e métricas para agentes.
- **Confident AI Docs: Metrics overview** — Categorias e tipos de métricas.
- **DeepEval Agentic metrics** — Métricas específicas para agentes.

---