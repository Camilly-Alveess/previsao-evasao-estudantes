# Previsão de Evasão de Estudantes

Pipeline de Machine Learning para prever a **evasão de estudantes** de uma faculdade a partir de dados de desempenho acadêmico, com deploy em Streamlit.
---

## 1. Objetivo

Construir um modelo de classificação binária que estime a probabilidade de um aluno **desistir do curso**, para apoiar ações de acolhimento e retenção antes que a evasão aconteça.

- **Alvo (`evasao`):** `1` = Desistente, `0` = Graduado.

## 2. Estrutura do repositório

```
├── app.py                      # aplicação Streamlit
├── outlier_clipper.py          # transformador personalizado usado no pipeline
├── requirements.txt            # dependências (scikit-learn fixado em 1.6.1)
├── Modelo_Evasao.ipynb         # EDA, pré-processamento, treino e avaliação
├── data/
│   ├── dados_treino.csv        # 2.782 alunos (80%)
│   └── dados_teste.csv         # 696 alunos (20%)
└── model/
    ├── modelo_evasao_pipeline.joblib   # pipeline usado pelo app
    └── modelo_evasao_final.joblib      # mesmo modelo, salvo como Pipeline
```

## 3. Dados

A base original tem **4.424 alunos e 28 colunas** (perfil, situação financeira, desempenho no 1º e no 2º semestre e indicadores macroeconômicos). A coluna `Target` tem três valores:

| Target | Alunos |
|---|---|
| Graduado | 2.209 |
| Desistente | 1.421 |
| Matriculado | 794 |

## 4. Passos desenvolvidos

### 4.1 Preparação e limpeza

1. **Escopo do problema:** os alunos `Matriculado` foram excluídos, porque o desfecho deles ainda é desconhecido e rotulá-los introduziria ruído no treino e na avaliação. Ficaram **3.630** alunos (Graduado e Desistente).
2. **Tipos:** colunas numéricas gravadas como texto com vírgula decimal (`TaxaDesemprego`, `PIB`, notas) foram convertidas para `float`.
3. **Registros inconsistentes:** foram identificados **152 alunos sem nenhum dado acadêmico** (zero disciplinas inscritas, creditadas, avaliadas e aprovadas nos dois semestres). Desses, 75 constam como graduados, o que é incompatível com a ausência total de registros. Eles foram removidos, resultando em **3.478 alunos** com **38,6% de evasão**.

### 4.2 Análise exploratória (EDA)

- **Situação financeira:** alunos devedores evadem **75,5%** das vezes, contra **34,5%** dos não devedores. `MensalidadesEmDia` tem correlação de cerca de -0,44 com a evasão e `Devedor`, de cerca de +0,27.
- **Desempenho acadêmico:** `UnidadesCurriculares2SemestreAprovado` (≈ -0,65) e `UnidadesCurriculares1SemestreAprovado` (≈ -0,55) são as variáveis mais correlacionadas com a evasão. Quanto mais disciplinas aprovadas, menor o risco.
- **Gênero:** a evasão é maior entre alunos do gênero masculino (acima de 50%) do que entre o feminino (cerca de 30%).
- **Curso:** a taxa varia muito, de mais de 80% em alguns cursos (como Tecnologias de Produção de Biocombustíveis e Engenharia Informática) a menos de 21% em outros (Enfermagem e Serviço Social).
- **Indicadores macroeconômicos** (`TaxaDesemprego`, `PIB`): correlação próxima de zero com a evasão individual, o que é esperado, já que variam por período e não por aluno.

Também foram exploradas variáveis derivadas: eficiência acadêmica por semestre (aprovadas ÷ inscritas), evolução da eficiência e das notas entre semestres, e uma flag de risco financeiro (devedor com mensalidade atrasada). A EDA mostrou que alunos que mantêm ou melhoram a eficiência quase não evadem.

### 4.3 Pré-processamento (feature engineering no pipeline)

Todo o tratamento está dentro de um `ColumnTransformer` do scikit-learn, ajustado apenas nos dados de treino, o que evita vazamento de dados (data leakage):

| Tipo | Etapas |
|---|---|
| Numéricas | imputação pela mediana → limitação de outliers (`OutlierClipper`, winsorização por 1,5 × IQR) → `StandardScaler` |
| Categóricas | imputação pela moda → `OneHotEncoder(handle_unknown="ignore")` |

### 4.4 Separação treino e teste

Divisão **80/20 estratificada** (`random_state=42`): 2.782 alunos para treino e 696 para teste, mantendo a proporção de evasão (38,6%) nos dois conjuntos. O teste só foi usado na avaliação final.

### 4.5 Modelo e validação cruzada

- **Modelo:** `RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)`, encapsulado junto com o pré-processamento num único `Pipeline`.
- **Validação cruzada:** `StratifiedKFold` com 5 partições, executada sobre o conjunto de treino.

## 5. Resultados

**Validação cruzada (5 folds, treino):** acurácia média de **91,95%** e F1-score médio de **88,81%**.

**Conjunto de teste (696 alunos):**

| Classe | Precisão | Recall | F1 |
|---|---|---|---|
| Graduado | 0,90 | 0,97 | 0,93 |
| Desistente | 0,94 | 0,84 | 0,88 |

Acurácia de **91,5%** e ROC-AUC de **0,96**.

Matriz de confusão (teste):

| | Previsto: Graduado | Previsto: Desistente |
|---|---|---|
| **Real: Graduado** | 412 | 15 |
| **Real: Desistente** | 44 | 225 |

O modelo acerta 94% dos alunos que sinaliza como risco de evasão (apenas 15 falsos alarmes) e identifica 84% de todos os que de fato evadem (44 passam despercebidos).

## 6. Análise de overfitting e underfitting

| Conjunto | Acurácia | ROC-AUC |
|---|---|---|
| Treino | 1,00 | 1,00 |
| Validação cruzada | 0,92 | — |
| Teste | 0,92 | 0,96 |

- **Overfitting leve:** o desempenho de 100% no treino mostra que a Random Forest, com árvores sem limite de profundidade, memoriza os dados vistos. Isso é típico desse algoritmo e não significa, por si só, que o modelo generalize mal.
- **Generalização boa:** a acurácia da validação cruzada (91,95%) e a do teste (91,5%) são praticamente iguais. Ou seja, não há queda ao passar para dados inéditos, e o gap está entre treino e dados novos, não entre validação e teste.
- **Sem underfitting:** as métricas em validação e teste são altas (F1 de 0,88 na classe minoritária e ROC-AUC de 0,96), então o modelo capturou os padrões relevantes dos dados.
- **Como reduzir o gap:** limitar `max_depth` ou aumentar `min_samples_leaf` tende a aproximar treino e validação.

## 7. Deploy

A aplicação em Streamlit carrega o pipeline salvo e oferece:

- **Previsão:** formulário com os dados do aluno (ou preenchimento a partir de um aluno real do conjunto de teste), que retorna a probabilidade de evasão e uma faixa de risco (baixo, médio ou alto).
- **Desempenho do modelo:** métricas de treino vs. teste, matriz de confusão e as variáveis mais importantes.
- **Sobre:** resumo do método.

### Como rodar localmente

```bash
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

> O `scikit-learn` está fixado em **1.6.1**, a mesma versão usada para salvar o modelo. Versões diferentes podem impedir o carregamento do arquivo `.joblib`.

## 8. Limitações e próximos passos

- **Notas dos semestres:** nas colunas `UnidadesCurriculares1SemestreGrau` e `UnidadesCurriculares2SemestreGrau`, cerca de 41% a 44% dos valores estão com escala inflada (da ordem de 10^16), provavelmente por erro de conversão de vírgula decimal. Corrigir a escala antes do split e retreinar deve melhorar a qualidade dessas variáveis.
- **Variáveis derivadas:** a eficiência acadêmica, a evolução entre semestres e a flag de risco financeiro foram exploradas na EDA, mas ainda não fazem parte do modelo salvo. Incorporá-las ao pipeline é um próximo passo natural.
- **Nota da qualificação anterior:** está sendo tratada como categórica (texto com vírgula decimal). Converter para numérica é mais adequado.
- **Momento da previsão:** o modelo usa dados dos dois semestres, e um aluno que já saiu no meio do 2º semestre naturalmente terá poucas aprovações. Uma versão que use só dados até o 1º semestre permitiria agir mais cedo.
- **Alunos matriculados:** ficaram fora do treino por não terem desfecho conhecido. As previsões para esse grupo são exploratórias e não foram validadas.
- **Uso responsável:** a probabilidade é uma estimativa estatística para apoiar ações de retenção, não uma decisão sobre o aluno.

## 9. Tecnologias

Python, pandas, NumPy, scikit-learn, joblib, matplotlib, seaborn e Streamlit.

*Autora: Camilly Alves*
