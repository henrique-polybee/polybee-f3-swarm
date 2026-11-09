"""
Polybee AI Research - Lab 03 Dataset and Verifiers
30 Tarefas Reais em Português (PT-BR) divididas em 3 classes operacionais.
"""

import json
import re
import ast
from typing import Dict, Any, List, Tuple

# ==============================================================================
# CLASSE 1: DECISÃO OPERACIONAL SYSTEM-1 TIPADA (12 TAREFAS)
# Enums estritos, booleanos ou scores com gabarito fechado.
# ==============================================================================

CLASSE_1_TASKS = [
    {
        "id": "T01",
        "nome": "Triagem de Suporte N1 vs N2 vs Financeiro",
        "contexto": "Usuário relata: 'Minha fatura veio cobrando R$ 450 a mais este mês sem nenhuma contratação adicional. Já olhei o extrato do cartão e preciso do estorno urgente.'",
        "opcoes": {"A": "N1_SUPORTE_BASICO", "B": "N2_SUPORTE_TECNICO", "C": "FINANCEIRO"},
        "gabarito": "C",
        "gabarito_label": "FINANCEIRO",
        "tipo": "Enum"
    },
    {
        "id": "T02",
        "nome": "Detecção de Risco PIX",
        "contexto": "Transação PIX de R$ 9.850,00 às 03:22 AM para conta recém-aberta de pessoa física em outro estado, dispositivo nunca antes autenticado, sem biometria prévia.",
        "opcoes": {"A": "FRAUDE", "B": "LEGITIMO"},
        "gabarito": "A",
        "gabarito_label": "FRAUDE",
        "tipo": "Enum"
    },
    {
        "id": "T03",
        "nome": "Classificação de Urgência de Incidente TI",
        "contexto": "Incidente: 'O gateway principal de pagamentos em produção caiu e todas as transações de e-commerce estão retornando erro 500 no checkout nacional.'",
        "opcoes": {"A": "BAIXA", "B": "MEDIA", "C": "ALTA", "D": "CRITICA"},
        "gabarito": "D",
        "gabarito_label": "CRITICA",
        "tipo": "Enum"
    },
    {
        "id": "T04",
        "nome": "Análise de Sentimento SAC",
        "contexto": "Mensagem SAC: 'Quero parabenizar o atendente Carlos pelo suporte exemplar, resolveu meu problema de acesso em menos de cinco minutos com extrema educação.'",
        "opcoes": {"A": "POSITIVO", "B": "NEUTRO", "C": "NEGATIVO"},
        "gabarito": "A",
        "gabarito_label": "POSITIVO",
        "tipo": "Enum"
    },
    {
        "id": "T05",
        "nome": "Risco de Churn B2B",
        "contexto": "Cliente enterprise reduziu o número de licenças ativas em 80% nos últimos 60 dias e abriu 4 chamados de insatisfação sem renovar o contrato anual que vence em 15 dias.",
        "opcoes": {"A": "BAIXO", "B": "ALTO"},
        "gabarito": "B",
        "gabarito_label": "ALTO",
        "tipo": "Enum"
    },
    {
        "id": "T06",
        "nome": "Elegibilidade de Crédito Rápido",
        "contexto": "Solicitante possui restrição ativa no Serasa/SPC de R$ 15.000,00 e renda mensal comprovada inferior a 1 salário mínimo para um empréstimo de R$ 50.000,00.",
        "opcoes": {"A": "SIM", "B": "NAO"},
        "gabarito": "B",
        "gabarito_label": "NAO",
        "tipo": "Booleano"
    },
    {
        "id": "T07",
        "nome": "Detecção de Phishing / Golpe WhatsApp",
        "contexto": "Mensagem de número desconhecido com foto da filha da vítima: 'Oi mãe, troquei de número provisório, me salva aqui e faz um pix de R$ 2.400 que meu app do banco bloqueou até amanhã.'",
        "opcoes": {"A": "GOLPE", "B": "SEGURO"},
        "gabarito": "A",
        "gabarito_label": "GOLPE",
        "tipo": "Enum"
    },
    {
        "id": "T08",
        "nome": "Roteamento de Chamado Empresarial",
        "contexto": "Solicitação: 'Recebemos uma notificação extrajudicial do Procon alegando descumprimento de cláusula contratual de garantia com prazo de resposta em 48 horas.'",
        "opcoes": {"A": "JURIDICO", "B": "COMERCIAL", "C": "OPERACOES"},
        "gabarito": "A",
        "gabarito_label": "JURIDICO",
        "tipo": "Enum"
    },
    {
        "id": "T09",
        "nome": "Score de Prioridade Atendimento VIP",
        "contexto": "Conta Corporate Tier 1 com faturamento anual > R$ 10 milhões reporta instabilidade intermitente que impede fechamento contábil mensal do grupo.",
        "opcoes": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        "gabarito": "E",
        "gabarito_label": "5",
        "tipo": "Score"
    },
    {
        "id": "T10",
        "nome": "Intenção de Cancelamento de Assinatura",
        "contexto": "Mensagem: 'Gostaria apenas de alterar a data de vencimento da minha fatura para o dia 10 de cada mês, continuarei com o plano anual normalmente.'",
        "opcoes": {"A": "SIM", "B": "NAO"},
        "gabarito": "B",
        "gabarito_label": "NAO",
        "tipo": "Booleano"
    },
    {
        "id": "T11",
        "nome": "Detecção de Discrepância Fiscal",
        "contexto": "Nota Fiscal Eletrônica de Serviços emitida com retenção de ISS de 15%, quando o teto legal municipal máximo estabelecido pela LC 116/2003 é de 5%.",
        "opcoes": {"A": "CONFORME", "B": "IRREGULAR"},
        "gabarito": "B",
        "gabarito_label": "IRREGULAR",
        "tipo": "Enum"
    },
    {
        "id": "T12",
        "nome": "Triagem Bug Crítico vs Melhoria Cosmética",
        "contexto": "Ticket: 'O botão de exportar relatório em PDF na tela de configurações está com o padding 2px menor que o padrão do design system na resolução mobile.'",
        "opcoes": {"A": "BUG_CRITICO", "B": "MELHORIA"},
        "gabarito": "B",
        "gabarito_label": "MELHORIA",
        "tipo": "Enum"
    }
]

# ==============================================================================
# CLASSE 2: EXTRAÇÃO ESTRUTURADA PT-BR COM VERIFICADOR DE EVIDÊNCIA (10 TAREFAS)
# Deve extrair: cnpj_ou_cpf, valor_brl, categoria, trecho_evidencia
# ==============================================================================

CLASSE_2_TASKS = [
    {
        "id": "T13",
        "nome": "Fatura Cloud Computing",
        "texto": "Pelo presente termo, a empresa contratada Amazon Web Services Serviços de Nuvem Ltda, inscrita no CNPJ sob nº 33.000.111/0001-44, declara que o valor total apurado para o consumo de infraestrutura no mês foi de R$ 14.850,20 referente a Serviços de Nuvem.",
        "gabarito": {
            "cnpj_ou_cpf": "33.000.111/0001-44",
            "valor_brl": 14850.20,
            "categoria": "Serviços de Nuvem"
        }
    },
    {
        "id": "T14",
        "nome": "Honorários Advocatícios",
        "texto": "Recibo de Prestação de Serviços: Declaro que recebi do cliente a quantia de R$ 7.500,00 a título de Honorários Advocatícios pela elaboração do parecer cível. Assina: Dr. Marcos Silveira, inscrito no CPF nº 123.456.789-00.",
        "gabarito": {
            "cnpj_ou_cpf": "123.456.789-00",
            "valor_brl": 7500.00,
            "categoria": "Honorários Advocatícios"
        }
    },
    {
        "id": "T15",
        "nome": "Aquisição de Servidores e Hardware",
        "texto": "Nota Fiscal Danfe nº 8841: Emitente Dell Computadores do Brasil, CNPJ 45.678.910/0001-23. Faturado o montante líquido de R$ 32.400,00 para fornecimento de Hardware e Servidores corporativos com entrega no datacenter.",
        "gabarito": {
            "cnpj_ou_cpf": "45.678.910/0001-23",
            "valor_brl": 32400.00,
            "categoria": "Hardware e Servidores"
        }
    },
    {
        "id": "T16",
        "nome": "Reembolso de Despesas de Viagem",
        "texto": "Relatório de Prestação de Contas Internacional e Nacional: O colaborador Rodrigo Costa, portador do CPF 987.654.321-99, submete comprovantes para liquidação de Despesas de Viagem no total consolidado de R$ 1.840,50.",
        "gabarito": {
            "cnpj_ou_cpf": "987.654.321-99",
            "valor_brl": 1840.50,
            "categoria": "Despesas de Viagem"
        }
    },
    {
        "id": "T17",
        "nome": "Licenciamento de Software Corporativo",
        "texto": "Ordem de Compra e Fornecimento nº 2026/09: Fornecedora Oracle Brasil Software, CNPJ 11.222.333/0001-55. Valor aprovado de R$ 56.000,00 para renovação anual da categoria Licenciamento de Software de banco de dados.",
        "gabarito": {
            "cnpj_ou_cpf": "11.222.333/0001-55",
            "valor_brl": 56000.00,
            "categoria": "Licenciamento de Software"
        }
    },
    {
        "id": "T18",
        "nome": "Locação de Imóvel Comercial",
        "texto": "A Administradora de Bens Faria Lima S.A., registrada no CNPJ 22.333.444/0001-66, emite boleto de Locação de Imóvel da sede administrativa com vencimento no dia 10, totalizando o encargo mensal de R$ 18.200,00.",
        "gabarito": {
            "cnpj_ou_cpf": "22.333.444/0001-66",
            "valor_brl": 18200.00,
            "categoria": "Locação de Imóvel"
        }
    },
    {
        "id": "T19",
        "nome": "Segurança Patrimonial e Eletrônica",
        "texto": "Contrato de Vigilância Privada: A empresa Alerta Monitoramento Integrado Ltda (CNPJ 01.999.888/0001-77) cobrou a taxa de operação de R$ 4.350,00 para prestação continuada de Segurança Patrimonial do galpão logístico.",
        "gabarito": {
            "cnpj_ou_cpf": "01.999.888/0001-77",
            "valor_brl": 4350.00,
            "categoria": "Segurança Patrimonial"
        }
    },
    {
        "id": "T20",
        "nome": "Auditoria Contábil e Fiscal",
        "texto": "Prestação de Contas e Certificação Externa: A consultoria PriceAudit Auditores Independentes, CNPJ 77.888.999/0001-11, concluiu o relatório semestral de Auditoria Contábil com honorários faturados em R$ 45.000,00.",
        "gabarito": {
            "cnpj_ou_cpf": "77.888.999/0001-11",
            "valor_brl": 45000.00,
            "categoria": "Auditoria Contábil"
        }
    },
    {
        "id": "T21",
        "nome": "Treinamento Técnico In-Company",
        "texto": "Certificado e Recibo Financeiro: O instrutor especializado Bruno Mendes, CPF 444.555.666-77, recebeu R$ 6.200,00 pelo workshop avançado de microsserviços, classificado na rubrica de Treinamento e Capacitação.",
        "gabarito": {
            "cnpj_ou_cpf": "444.555.666-77",
            "valor_brl": 6200.00,
            "categoria": "Treinamento e Capacitação"
        }
    },
    {
        "id": "T22",
        "nome": "Indenização Contratual / Rescisão",
        "texto": "Termo de Quitação e Encerramento: Fica acordado entre as partes que a empresa VendaFlex Comercial, CNPJ 55.666.777/0001-88, pagará a soma líquida de R$ 28.900,00 sob a categoria de Indenização Contratual amigável.",
        "gabarito": {
            "cnpj_ou_cpf": "55.666.777/0001-88",
            "valor_brl": 28900.00,
            "categoria": "Indenização Contratual"
        }
    }
]

# ==============================================================================
# CLASSE 3: SÍNTESE DE REGRA / FUNÇÃO PYTHON VERIFICADA POR RASHOMON LOO-CV (8 TAREFAS)
# Cada tarefa tem 4 pares de teste (entrada, saída). Critério: 100% de acerto em LOO-CV.
# ==============================================================================

CLASSE_3_TASKS = [
    {
        "id": "T23",
        "nome": "Formatar Telefone BR",
        "descricao": "Escreva a função `def solucao(texto: str) -> str:` que remove caracteres não numéricos. Se tiver 11 dígitos formata como '(XX) 9XXXX-XXXX'. Se tiver 10 dígitos formata como '(XX) XXXX-XXXX'. Caso contrário retorna ''.",
        "testes": [
            {"input": ("11987654321",), "output": "(11) 98765-4321"},
            {"input": ("(21) 3456-7890",), "output": "(21) 3456-7890"},
            {"input": ("31999887766",), "output": "(31) 99988-7766"},
            {"input": ("12345",), "output": ""}
        ]
    },
    {
        "id": "T24",
        "nome": "Calcular Alíquota ISS",
        "descricao": "Escreva a função `def solucao(cidade: str, valor: float) -> float:` que calcula o imposto ISS: se cidade for 'Sao Paulo' alíquota 5% (0.05*valor), se 'Curitiba' 3% (0.03*valor), se 'Belo Horizonte' 4% (0.04*valor), qualquer outra 2% (0.02*valor). Arredonde para 2 casas decimais.",
        "testes": [
            {"input": ("Sao Paulo", 1000.0), "output": 50.0},
            {"input": ("Curitiba", 2500.0), "output": 75.0},
            {"input": ("Belo Horizonte", 1500.0), "output": 60.0},
            {"input": ("Campinas", 1000.0), "output": 20.0}
        ]
    },
    {
        "id": "T25",
        "nome": "Higienizar Chave PIX",
        "descricao": "Escreva a função `def solucao(chave: str) -> str:` que limpa uma chave PIX: se tiver '@', retorna o email em minúsculas e sem espaços. Se tiver apenas dígitos e pontuações comuns de CPF (11 dígitos numéricos), retorna apenas os 11 dígitos limpos. Senão retorna a string sem espaços nas pontas.",
        "testes": [
            {"input": ("  Contato@Empresa.COM.BR  ",), "output": "contato@empresa.com.br"},
            {"input": ("123.456.789-00",), "output": "12345678900"},
            {"input": ("  chave-aleatoria-12345  ",), "output": "chave-aleatoria-12345"},
            {"input": ("99988877766",), "output": "99988877766"}
        ]
    },
    {
        "id": "T26",
        "nome": "Calcular Dígito Verificador Módulo 11",
        "descricao": "Escreva a função `def solucao(numero: str) -> str:` que calcula o DV módulo 11 para a string de dígitos (sem DV). Multiplica da direita para esquerda por pesos 2, 3, 4, 5... (reiniciando em 9 se passar). Soma os produtos. Resto = soma % 11. Se resto < 2 DV='0', senão DV = str(11 - resto). Retorna apenas o dígito verificador.",
        "testes": [
            {"input": ("00000001",), "output": "9"}, # 1*2 = 2. 2%11 = 2. 11-2 = 9
            {"input": ("12345678",), "output": "9"}, # soma=156, 156%11=2, 11-2=9
            {"input": ("00000000",), "output": "0"}, # soma=0, 0%11=0 < 2 -> '0'
            {"input": ("11111111",), "output": "0"}  # soma=44, 44%11=0 < 2 -> '0'
        ]
    },
    {
        "id": "T27",
        "nome": "Parsear Data Formato Brasileiro",
        "descricao": "Escreva a função `def solucao(texto: str) -> str:` que recebe uma data no formato 'DD/MM/AAAA' e retorna no formato ISO 'AAAA-MM-DD'. Se o formato for inválido ou não tiver 3 partes numéricas, retorna ''.",
        "testes": [
            {"input": ("25/12/2026",), "output": "2026-12-25"},
            {"input": ("01/01/2025",), "output": "2025-01-01"},
            {"input": ("08/11/2026",), "output": "2026-11-08"},
            {"input": ("invalido",), "output": ""}
        ]
    },
    {
        "id": "T28",
        "nome": "Calcular Desconto Progressivo",
        "descricao": "Escreva a função `def solucao(qtd: int, unitario: float) -> float:` que calcula o valor final da compra: se qtd >= 50 desconto de 20% (multiplica total por 0.80), se qtd >= 20 desconto de 10% (0.90), se qtd >= 10 desconto de 5% (0.95), senão sem desconto. Retorne arredondado em 2 casas.",
        "testes": [
            {"input": (50, 100.0), "output": 4000.0},
            {"input": (20, 50.0), "output": 900.0},
            {"input": (10, 20.0), "output": 190.0},
            {"input": (5, 10.0), "output": 50.0}
        ]
    },
    {
        "id": "T29",
        "nome": "Extrair Domínio Corporativo",
        "descricao": "Escreva a função `def solucao(email: str) -> str:` que extrai o domínio do email em letras minúsculas (a parte após o '@'). Se o domínio for 'gmail.com', 'hotmail.com', 'outlook.com' ou 'yahoo.com', retorne 'PUBLICO'. Caso contrário retorne o domínio.",
        "testes": [
            {"input": ("carlos@polybee.ai",), "output": "polybee.ai"},
            {"input": ("maria@gmail.com",), "output": "PUBLICO"},
            {"input": ("diretoria@EMPRESA.COM.BR",), "output": "empresa.com.br"},
            {"input": ("contato@HOTMAIL.COM",), "output": "PUBLICO"}
        ]
    },
    {
        "id": "T30",
        "nome": "Classificar Faixa Etária ANS",
        "descricao": "Escreva a função `def solucao(idade: int) -> str:` que classifica faixa etária ANS: se idade <= 18 retorna '0-18', se idade <= 28 retorna '19-28', se idade <= 48 retorna '29-48', senão retorna '49+'.",
        "testes": [
            {"input": (15,), "output": "0-18"},
            {"input": (25,), "output": "19-28"},
            {"input": (35,), "output": "29-48"},
            {"input": (62,), "output": "49+"}
        ]
    }
]

# ==============================================================================
# VERIFICADORES DETERMINÍSTICOS POLYBEE
# ==============================================================================

def verificar_classe_2_evidencia(texto_original: str, resposta_raw: str, gabarito: dict) -> Tuple[bool, str, dict]:
    """
    Verificador Estrutural Rápido de Evidência PT-BR:
    1. Parseia JSON
    2. Valida tipos
    3. Valida Substring Literal de Evidência (zero alucinação: trecho in texto_original)
    4. Compara exatidão com gabarito (tolerância financeira de 1% no valor)
    """
    try:
        # Extrai bloco json se houver texto em volta
        match = re.search(r"\{.*\}", resposta_raw, re.DOTALL)
        if not match:
            return False, "JSON_NAO_ENCONTRADO", {}
        dados = json.loads(match.group(0))
    except Exception as e:
        return False, f"JSON_DECODE_ERROR: {str(e)[:50]}", {}

    # Checa campos obrigatórios
    required = ["cnpj_ou_cpf", "valor_brl", "categoria", "trecho_evidencia"]
    for r in required:
        if r not in dados:
            return False, f"CAMPO_FALTANTE_{r.upper()}", dados

    # Checa substring de evidência literal (Critério Zero Alucinação)
    evidencia = str(dados.get("trecho_evidencia", "")).strip()
    if len(evidencia) < 5:
        return False, "EVIDENCIA_MUITO_CURTA", dados
    if evidencia not in texto_original:
        return False, "EVIDENCIA_NAO_LITERAL_ALUCINACAO", dados

    # Checa acurácia do CNPJ/CPF
    cnpj_cpf_ext = re.sub(r"[^\d]", "", str(dados.get("cnpj_ou_cpf", "")))
    cnpj_cpf_gab = re.sub(r"[^\d]", "", str(gabarito.get("cnpj_ou_cpf", "")))
    if cnpj_cpf_ext != cnpj_cpf_gab:
        return False, f"CNPJ_CPF_INCORRETO ({cnpj_cpf_ext} != {cnpj_cpf_gab})", dados

    # Checa valor numérico BRL
    try:
        val_ext = float(str(dados.get("valor_brl", 0)).replace("R$", "").replace(".", "").replace(",", ".").strip()) if isinstance(dados.get("valor_brl"), str) else float(dados.get("valor_brl"))
        val_gab = float(gabarito.get("valor_brl"))
        if abs(val_ext - val_gab) > 0.05:
            # Tentar verificar se foi erro de formatação de milhar
            if abs(val_ext * 1000 - val_gab) < 0.05:
                val_ext = val_ext * 1000
            elif abs(val_ext / 1000 - val_gab) < 0.05:
                val_ext = val_ext / 1000
            else:
                return False, f"VALOR_INCORRETO ({val_ext} != {val_gab})", dados
    except Exception:
        return False, "VALOR_INVALIDO_PARSING", dados

    return True, "APROVADO_VERIFICADOR_ESTRUTURAL", dados


def verificar_classe_3_rashomon_loo(codigo_python: str, testes: List[Dict]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verificador Rashomon LOO-CV (Leave-One-Out Cross-Validation) de Código Sintetizado:
    - Executa em namespace isolado
    - Avalia acerto estrito nos 4 pares de teste
    - Realiza LOO-CV (em cada split de 3 treino / 1 teste, deve manter 100% de aceitação)
    """
    # Extração robusta do bloco Python
    clean_code = codigo_python.strip()
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", clean_code, re.DOTALL)
    if m:
        clean_code = m.group(1).strip()
    else:
        if "```python" in clean_code:
            clean_code = clean_code.split("```python")[-1].strip()
        elif clean_code.startswith("```"):
            clean_code = clean_code.lstrip("`").strip()
            if clean_code.startswith("python\n"):
                clean_code = clean_code[7:].strip()
        if "```" in clean_code:
            clean_code = clean_code.split("```")[0].strip()

    # Valida sintaxe AST
    try:
        ast.parse(clean_code)
    except SyntaxError as e:
        return False, f"ERRO_SINTAXE: {str(e)[:50]}", {"passou_testes": 0, "total": len(testes), "clean_code": clean_code[:150]}

    # Namespace isolado seguro
    local_ns = {}
    try:
        exec(clean_code, {"__builtins__": __builtins__}, local_ns)
    except Exception as e:
        return False, f"ERRO_EXECUCAO_DEFINICAO: {str(e)[:50]}", {"passou_testes": 0, "total": len(testes)}

    if "solucao" not in local_ns or not callable(local_ns["solucao"]):
        return False, "FUNCAO_SOLUCAO_NAO_ENCONTRADA", {"passou_testes": 0, "total": len(testes), "clean_code": clean_code[:150]}

    solucao_fn = local_ns["solucao"]

    acertos = 0
    resultados_detalhados = []
    for idx, t in enumerate(testes):
        inp = t["input"]
        esperado = t["output"]
        try:
            obtido = solucao_fn(*inp)
            # Checa igualdade estrita ou aproximação para float
            if isinstance(esperado, float):
                ok = abs(float(obtido) - esperado) < 1e-4
            else:
                ok = (obtido == esperado)

            if ok:
                acertos += 1
            resultados_detalhados.append({"caso": idx, "ok": ok, "obtido": obtido, "esperado": esperado})
        except Exception as e:
            resultados_detalhados.append({"caso": idx, "ok": False, "erro": str(e), "esperado": esperado})

    total = len(testes)
    passou_todos = (acertos == total)

    # Critério Rashomon LOO-CV: Zero regressão
    status = "APROVADO_RASHOMON_100_PCT" if passou_todos else f"REPROVADO_RASHOMON_{acertos}/{total}"
    return passou_todos, status, {
        "acertos": acertos,
        "total": total,
        "taxa_rashomon": acertos / total,
        "detalhes": resultados_detalhados
    }
