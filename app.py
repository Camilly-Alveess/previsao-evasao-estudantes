"""App Streamlit: previsão de evasão de estudantes.

Arquivos necessários (em qualquer pasta do projeto):
    modelo_evasao_pipeline.joblib, dados_treino.csv, dados_teste.csv, outlier_clipper.py
Rodar localmente:  streamlit run app.py
"""
import os
from pathlib import Path

import __main__
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from outlier_clipper import OutlierClipper

# O pipeline foi salvo no notebook, onde OutlierClipper vivia em __main__.
# Registramos a classe lá para o joblib conseguir reconstruir o modelo.
__main__.OutlierClipper = OutlierClipper

st.set_page_config(page_title="Risco de evasão", page_icon="🎓", layout="wide")

BASE = Path(__file__).parent

CAT_COLS = ["EstadoCivil", "Curso", "QualificacaoAnterior", "Nacionalidade", "Genero"]
BIN_COLS = ["NecessidadesEspeciais", "Devedor", "MensalidadesEmDia", "Bolsista", "International"]
GRAU_COLS = [f"UnidadesCurriculares{s}SemestreGrau" for s in (1, 2)]
INT_COLS = [f"UnidadesCurriculares{s}Semestre{c}" for s in (1, 2)
            for c in ("Creditado", "Inscrito", "Avaliacoes", "Aprovado", "SemAvaliacoes")]

ROTULOS = {
    "Curso": "Curso", "EstadoCivil": "Estado civil", "Nacionalidade": "Nacionalidade",
    "Genero": "Gênero", "QualificacaoAnterior": "Qualificação anterior",
    "QualificacaoAnteriorGrau": "Nota da qualificação anterior",
    "NotaAdmissao": "Nota de admissão", "NecessidadesEspeciais": "Necessidades especiais?",
    "International": "Aluno internacional?", "Devedor": "Devedor?",
    "MensalidadesEmDia": "Mensalidades em dia?", "Bolsista": "Bolsista?",
    "TaxaDesemprego": "Taxa de desemprego (%)", "TaxaInflacao": "Taxa de inflação (%)",
    "PIB": "PIB (variação %)",
}
NENHUM = "— preencher manualmente —"


def listar_arquivos():
    """Percorre o repositório ignorando pastas ocultas e ambientes virtuais."""
    for raiz, pastas, arquivos in os.walk(BASE):
        pastas[:] = [d for d in pastas if not d.startswith(".")
                     and d not in ("__pycache__", "venv", "site-packages", "node_modules")]
        for nome in arquivos:
            yield Path(raiz) / nome


def achar(*nomes):
    """Procura o arquivo em qualquer pasta do projeto, sem diferenciar maiúsculas/minúsculas.
    Aceita mais de um nome, em ordem de preferência."""
    alvos = [n.lower() for n in nomes]
    todos = list(listar_arquivos())
    achados = [p for p in todos if p.name.lower() in alvos]
    if achados:
        achados.sort(key=lambda p: (alvos.index(p.name.lower()), len(p.parts)))
        return achados[0]
    st.error(f"Não encontrei o arquivo `{nomes[0]}`. Estes são os arquivos que o app enxerga:")
    st.code("\n".join(sorted(str(p.relative_to(BASE)) for p in todos)[:80]))
    st.stop()


CAMINHO_MODELO = achar("modelo_evasao_pipeline.joblib", "modelo_evasao_final.joblib")
CAMINHO_TREINO = achar("dados_treino.csv")
CAMINHO_TESTE = achar("dados_teste.csv")


@st.cache_resource
def carregar_modelo(caminho):
    obj = joblib.load(caminho)
    if isinstance(obj, dict):
        return obj["model_pipeline"], list(obj["input_features"])
    return obj, list(obj.feature_names_in_)


@st.cache_data
def carregar_dados(caminho_treino, caminho_teste):
    return pd.read_csv(caminho_treino), pd.read_csv(caminho_teste)


def fmt_nota(v):
    """Formato esperado pelo modelo para QualificacaoAnteriorGrau (texto, vírgula decimal)."""
    v = round(float(v), 1)
    return str(int(v)) if v.is_integer() else str(v).replace(".", ",")


def padrao(col, treino):
    s = treino[col]
    if col == "QualificacaoAnteriorGrau":
        return float(pd.to_numeric(s.astype(str).str.replace(",", ".")).median())
    if col in GRAU_COLS:
        return float(s[s <= 20].median())
    if col in CAT_COLS:
        return s.mode()[0]
    if col in BIN_COLS:
        return int(s.mode()[0])
    if col in INT_COLS:
        return int(s.median())
    return float(s.median())


def valor_exemplo(col, v):
    if col == "QualificacaoAnteriorGrau":
        return float(str(v).replace(",", "."))
    if col in GRAU_COLS:
        v = float(v)
        while v > 20:  # corrige notas gravadas com escala inflada
            v /= 10
        return v
    if col in CAT_COLS:
        return v
    if col in BIN_COLS or col in INT_COLS:
        return int(v)
    return float(v)


modelo, colunas = carregar_modelo(str(CAMINHO_MODELO))
treino, teste = carregar_dados(str(CAMINHO_TREINO), str(CAMINHO_TESTE))

# Estado inicial dos widgets (valores típicos da base de treino)
for c in colunas:
    st.session_state.setdefault(c, padrao(c, treino))

exemplos = {
    f"Aluno #{i} (real: {'Desistente' if y == 1 else 'Graduado'})": i
    for i, y in enumerate(teste["evasao"].head(40))
}


def aplicar_exemplo():
    escolha = st.session_state["exemplo"]
    if escolha == NENHUM:
        return
    linha = teste.iloc[exemplos[escolha]]
    for c in colunas:
        st.session_state[c] = valor_exemplo(c, linha[c])


def campo_cat(col, container=st):
    container.selectbox(ROTULOS[col], sorted(treino[col].unique()), key=col)


def campo_bin(col, container=st):
    container.selectbox(ROTULOS[col], [0, 1], key=col,
                        format_func=lambda v: "Sim" if v == 1 else "Não")


def campo_float(col, rotulo, container=st, minimo=0.0, maximo=200.0):
    container.number_input(rotulo, min_value=minimo, max_value=maximo, step=0.1, key=col)


def campo_int(col, rotulo, container=st):
    container.number_input(rotulo, min_value=0, max_value=60, step=1, key=col)


st.title("🎓 Previsão de evasão de estudantes")
st.caption("Modelo Random Forest treinado com dados de desempenho acadêmico. "
           "Informe os dados do aluno e veja a probabilidade estimada de evasão.")

aba_prev, aba_perf, aba_sobre = st.tabs(["🔮 Previsão", "📊 Desempenho do modelo", "ℹ️ Sobre"])

# ---------------------------------------------------------------- Previsão
with aba_prev:
    st.selectbox("Preencher com um aluno do conjunto de teste (opcional)",
                 [NENHUM] + list(exemplos), key="exemplo", on_change=aplicar_exemplo)
    resultado = st.container()

    st.subheader("Perfil do aluno")
    a, b, c = st.columns(3)
    campo_cat("Curso", a)
    campo_cat("EstadoCivil", a)
    campo_cat("Nacionalidade", a)
    campo_cat("Genero", b)
    campo_cat("QualificacaoAnterior", b)
    campo_float("QualificacaoAnteriorGrau", ROTULOS["QualificacaoAnteriorGrau"], b)
    campo_float("NotaAdmissao", ROTULOS["NotaAdmissao"], c)
    campo_bin("NecessidadesEspeciais", c)
    campo_bin("International", c)

    st.subheader("Situação financeira")
    a, b, c = st.columns(3)
    campo_bin("Devedor", a)
    campo_bin("MensalidadesEmDia", b)
    campo_bin("Bolsista", c)

    for s in (1, 2):
        st.subheader(f"Desempenho no {s}º semestre")
        p = f"UnidadesCurriculares{s}Semestre"
        a, b, c, d = st.columns(4)
        campo_int(f"{p}Inscrito", "Disciplinas inscritas", a)
        campo_int(f"{p}Aprovado", "Disciplinas aprovadas", b)
        campo_int(f"{p}Avaliacoes", "Avaliações realizadas", c)
        campo_float(f"{p}Grau", "Nota média (0 a 20)", d, 0.0, 20.0)

    with st.expander("Campos adicionais (preenchidos com valores típicos da base)"):
        a, b = st.columns(2)
        for s, col in ((1, a), (2, b)):
            p = f"UnidadesCurriculares{s}Semestre"
            campo_int(f"{p}Creditado", f"Disciplinas creditadas ({s}º sem.)", col)
            campo_int(f"{p}SemAvaliacoes", f"Disciplinas sem avaliações ({s}º sem.)", col)
        a, b, c = st.columns(3)
        campo_float("TaxaDesemprego", ROTULOS["TaxaDesemprego"], a, -20.0, 100.0)
        campo_float("TaxaInflacao", ROTULOS["TaxaInflacao"], b, -20.0, 100.0)
        campo_float("PIB", ROTULOS["PIB"], c, -20.0, 100.0)

    # Monta a entrada na ordem e no formato que o modelo espera
    entrada = {col: st.session_state[col] for col in colunas}
    entrada["QualificacaoAnteriorGrau"] = fmt_nota(entrada["QualificacaoAnteriorGrau"])
    df_in = pd.DataFrame([entrada])[colunas]
    prob = float(modelo.predict_proba(df_in)[0, 1])

    with resultado:
        st.divider()
        m1, m2 = st.columns([1, 2])
        m1.metric("Probabilidade de evasão", f"{prob:.1%}")
        previsao = "Desistente" if prob >= 0.5 else "Graduado"
        faixa = "baixo" if prob < 0.3 else "médio" if prob < 0.6 else "alto"
        m2.progress(min(max(prob, 0.0), 1.0))
        msg = f"Risco **{faixa}**. Classe prevista pelo modelo: **{previsao}**."
        (m2.success if faixa == "baixo" else m2.warning if faixa == "médio" else m2.error)(msg)
        for s in (1, 2):
            p = f"UnidadesCurriculares{s}Semestre"
            if st.session_state[f"{p}Aprovado"] > st.session_state[f"{p}Inscrito"]:
                st.warning(f"No {s}º semestre há mais disciplinas aprovadas do que inscritas. "
                           "Confira os valores.")
        st.divider()


# ------------------------------------------------------------- Desempenho
@st.cache_data
def calcular_desempenho():
    linhas, cm_teste = {}, None
    for nome, df in (("Treino", treino), ("Teste", teste)):
        y = df["evasao"]
        proba = modelo.predict_proba(df[colunas])[:, 1]
        pred = (proba >= 0.5).astype(int)
        linhas[nome] = {
            "Acurácia": accuracy_score(y, pred),
            "Precisão (evasão)": precision_score(y, pred),
            "Recall (evasão)": recall_score(y, pred),
            "F1 (evasão)": f1_score(y, pred),
            "ROC-AUC": roc_auc_score(y, proba),
        }
        if nome == "Teste":
            cm_teste = pd.DataFrame(
                confusion_matrix(y, pred),
                index=["Real: Graduado", "Real: Desistente"],
                columns=["Previsto: Graduado", "Previsto: Desistente"])

    pre = modelo.named_steps["preprocessor"]
    num_cols = list(pre.transformers_[0][2])
    cat_cols = list(pre.transformers_[1][2])
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    nomes = num_cols + list(ohe.get_feature_names_out(cat_cols))
    imp = pd.Series(modelo.named_steps["classifier"].feature_importances_, index=nomes)

    def origem(n):
        for col in sorted(cat_cols, key=len, reverse=True):
            if n.startswith(col + "_"):
                return col
        return n

    imp = imp.groupby(origem).sum().sort_values(ascending=False)
    return pd.DataFrame(linhas).T, cm_teste, imp


with aba_perf:
    metricas, cm, imp = calcular_desempenho()
    st.subheader("Métricas: treino vs. teste")
    st.dataframe(metricas.style.format("{:.3f}"))
    st.caption("A diferença entre treino e teste indica o quanto o modelo se ajustou aos "
               "dados vistos no treinamento (sinal de sobreajuste).")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Matriz de confusão (teste)")
        st.dataframe(cm)
    with c2:
        st.subheader("Variáveis mais importantes")
        st.bar_chart(imp.head(10).sort_values())

# ------------------------------------------------------------------ Sobre
with aba_sobre:
    st.markdown(
        """
**Como funciona**

- Alvo binário: `Desistente = 1` (evasão) e `Graduado = 0`. Alunos *Matriculados* ficaram
  de fora do treino por não terem desfecho conhecido.
- Pré-processamento dentro do pipeline: imputação (mediana/moda), limitação de outliers
  (IQR), padronização das numéricas e one-hot das categóricas.
- Modelo: Random Forest com `class_weight="balanced"`, avaliado com validação cruzada
  estratificada (5 folds) e depois num conjunto de teste separado.

**Uso responsável:** a probabilidade é uma estimativa estatística para apoiar ações de
acolhimento e retenção, não uma decisão sobre o aluno.
        """
    )
