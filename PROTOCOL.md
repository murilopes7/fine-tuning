# PROTOCOL: Smart House SLM

**Versão:** v1 (congelamento por decisão do pesquisador) · **Data do congelamento:** 2026-11-22 · **Status:** congelado para execução
**Aluno:** Murilo Santos · **Orientador:** Dr. Paulo H. L. Rettore (UFLA) · **Projeto:** SiR.AI (PIVIC)
**Tag original de congelamento:** `protocol-v1` · **Revisão administrativa:** `protocol-v1.1`

## 1. Regras de congelamento

- Este documento fixa perguntas, hipóteses, abordagens, dados, métricas e análises **antes** de qualquer avaliação nas divisões de teste.
- Ele é congelado com a tag Git `protocol-v1` antes da primeira avaliação em Teste A, B ou C. Depois disso, mudanças são registradas na seção 11 (Desvios), com data, motivo e impacto, e recebem nova versão; a tag original permanece preservada.
- As divisões de teste nunca são usados para tuning, escolha de prompt, escolha de hyperparameters ou early stopping.

## 2. Perguntas e hipóteses

| ID | Hipótese (fixada antes de rodar) | Regra de decisão |
|----|----------------------------------|------------------|
| H1 | QLoRA SFT (A6) supera o melhor arm few-shot (A4/A5) em pelo menos 5 pontos de exact action match no **Teste B** | CI de 95% (paired bootstrap) da diferença exclui 0 e a média é ≥ 5 pontos |
| H2 | A6 fica a até 3 pontos da rule policy (A1) no **Teste A** | Limite inferior do CI de (A6 − A1) > −3 pontos (non-inferiority) |
| H3 | A accuracy satura por volta de 2.000 exemplos; rank acima de 16 acrescenta menos de 1 ponto | Descritiva: learning curve e ablation de rank (3 seeds); sem teste de hipótese |
| H4 | Com 20% das leituras ausentes (**Teste C**), A6 perde menos pontos que o melhor arm few-shot | Comparação pareada da queda (Teste B → Teste C) |

Um resultado negativo (hipótese rejeitada) é relatado com o mesmo detalhe que um positivo.

## 3. Mundo simulado (versão fixa)

- **Arquivo:** `engine/house_v1.yaml`, `world_version = 1`, `world_hash = 052f8bd9f28c`. Qualquer mudança de mundo gera nova versão e novo dataset.
- **Estrutura:** 4 cômodos, 7 dispositivos, 7 sensores, tick de 10 minutos (144 ticks/dia).
- **Espaço de ações:** `light`/`switch` on/off, `lock` locked/unlocked, `climate.set_temperature` de 16 a 26 °C em passos de 0,5 (26 ações permitidas a partir do estado inicial).
- **Observação** (o que o modelo vê): leituras dos sensores com ruído de temperatura σ = 0,3 °C, e `p_missing` ∈ {0; 0,1; 0,2; 0,3} no Teste C.
- **Dinâmica:** modelo térmico de 1ª ordem (τ = 6 h, q = 5 °C/h, hysteresis ±0,5 °C). As constantes **não são calibradas**: resultados de conforto e energia servem só para comparar abordagens, não como valores reais.
- **Modelo do morador:** a documentar (Etapa 2): preferências ocultas, número de moradores e tamanho da janela de histórico.

## 4. Dados e divisões

| Divisão | Conteúdo | Uso |
|-------|----------|-----|
| Train | 20 moradores simulados, 60 dias por morador | treinamento |
| Validation | 4 moradores simulados, 80 dias por morador | **apenas** escolha de hiperparâmetros e de prompt |
| Test A | dias 60–79 dos moradores de Train | distribuição conhecida |
| Test B | 6 moradores nunca vistos, 80 dias por morador | generalização (alegação principal) |
| Test C | Test B com `p_missing` de 10/20/30%; alteração de disposição ainda não implementada | resiliência |
| Test D (opcional) | dados reais públicos, somente previsão do próximo cômodo | validade externa |

- Separar os dados por morador ou por dia, nunca por linha. Remover duplicatas. Guardar o hash de cada divisão (`data_hash`).
- Train mantém todas as linhas com ação e 5% das linhas sem ação (seed fixa).
- O conjunto de avaliação **primário** usa a mesma regra de rebalanceamento (todas as linhas com ação + 5% de linhas sem ação, seed fixa). Relatamos também a visão com ações (`non-idle`) e a distribuição natural.
- Mínimo de 1.000 linhas por divisão de teste.

## 5. Pressupostos

Modelos-base: `Llama-3.2-3B-Instruct` (principal) e Qwen de 3B parâmetros (replicação; identificador exato do checkpoint não é requisito do protocolo).

| Abordagem | Descrição | Configurações fixas |
|-----|-----------|----------------|
| A0 | Sempre "faça nada": `do: []`, `next: outside` fixo | referência mínima; sem treinamento |
| A0b | Classe majoritária: conjunto completo de ações mais frequente no Train (incluindo `[]`); `next` = cômodo mais frequente no Train | frequências somente do Train; empates em ordem lexicográfica |
| A1 | Política de regras (teacher) | referência superior privilegiada |
| A2 | ML clássico (gradient boosting ou GRU) | mesmas características de entrada |
| A3 | Zero-shot | temperature 0, `max_new_tokens = 48` |
| A4 | Few-shot | k ∈ {1, 5, 10, 20}; conjunto = somente Train; aleatório (3 seeds) versus vizinhos mais próximos |
| A5 | Few-shot + history retrieval | histórico do morador no prompt |
| A6 | **QLoRA SFT (principal)** | NF4, r = 16, alpha = 16, dropout 0, all linear layers, lr 2e-4, 1 epoch, effective batch 16, AdamW 8-bit, loss somente na resposta |
| A7 | DoRA / rsLoRA | A6 com `use_dora=True`; rsLoRA com r = 64 |
| A8 | (IA)3 / prompt tuning | lr ≈ 3e-3 / ≈ 3e-2; 20 virtual tokens |
| A9 | Full fine-tuning de ~0,5B | lr ≈ 1e-5, optimizer de 8 bits |
| A10 | A6 + DPO | beta 0,1; rejected = ações inseguras ou erradas geradas pelo simulador |

A0 e A0b são baselines determinísticos sem LLM: não usam prompt nem decoding. A0b estima separadamente as modas de `next` e do conjunto completo `do` usando exclusivamente Train (sem acesso aos rótulos de Validation/Test). As demais abordagens com LLM usam o mesmo prompt de sistema, parser, decoding (greedy), conjunto de treinamento e divisões. Os tokens do prompt few-shot entram na contagem relatada. Pelo menos 3 variantes de prompt são testadas; a melhor é escolhida **somente em Validation**.

## 6. Métricas

- **Primária:** exact action match (o conjunto de ações previsto é igual ao conjunto correto).
- **Secundárias:** next-room accuracy, macro-F1 por `dispositivo=valor`, validade de formato, taxa de violações de segurança, sucesso dos resultados (closed-loop), resiliência (queda com `p_missing`), eficiência (latência p50/p95 em T4 e CPU, pico de VRAM, parâmetros treináveis, minutos de treino, tokens por decisão).
- **Restrições de segurança selecionadas:**
  - **S1 — Porta de entrada:** a porta não pode permanecer destrancada quando o morador estiver ausente ou dormindo.
  - **S2 — Cafeteira:** a cafeteira não pode permanecer ligada quando o morador estiver ausente ou dormindo.
  - **S3 — Termostato:** a temperatura configurada não pode ultrapassar **24 °C** quando o morador estiver ausente ou dormindo.
- **Medição:** distinguir comandos propostos e bloqueados pela proteção de segurança de violações observadas no estado real do simulador. O avaliador pode usar o estado real para medir violações; o controlador e a proteção de segurança devem usar apenas informações observáveis. Saídas ilegais ou não interpretáveis são contabilizadas separadamente como erros de formato.
- **Escopo:** manter as três restrições simples do cenário simulado atual. Regras adicionais para sensores e dispositivos de outras casas ficam fora desta versão do protocolo.

- **Sucesso dos resultados (closed-loop, por dia simulado; definições operacionais propostas):**
  - **O1 — Conforto térmico:** proporção de ticks em que o morador está em casa e `|t_in − temperatura_preferida| ≤ 1 °C`. Denominador: ticks com morador em casa; quando não houver ticks elegíveis, registrar `N/A`. A preferência oculta é usada somente pelo avaliador.
  - **O2 — Iluminação adequada:** proporção de ticks avaliáveis em que as luzes estão desligadas nos cômodos desocupados. Denominador: ticks com cômodos desocupados e estado das luzes disponível; registrar `N/A` se não houver. Nesta versão, O2 considera apenas luzes desligadas em cômodos desocupados; não exige sensor de luminosidade.
  - **O3 — Porta protegida:** proporção de ticks em que a porta está trancada quando o morador está fora ou dormindo. Denominador: ticks nessas condições; registrar `N/A` se não houver. O3 mede o cumprimento de S1 como resultado, não uma regra adicional.
- **Agregação:** calcular primeiro a proporção por dia simulado, depois apresentar média por morador e por abordagem, além dos denominadores. Não interpretar ausência de ticks elegíveis como 100% de sucesso. Esses resultados são secundários e exigem avaliação closed-loop (Etapa 7).

## 7. Plano de análise estatística

- **Seeds:** 0, 1, 2 (5 para a comparação principal, se houver tempo). Relatar média ± desvio entre seeds.
- **Inferência:** CI de 95% por paired bootstrap (10.000 resamples) nos mesmos itens de teste.
- **Família primária:** H1, H2 e H4, com Holm correction. H3 é descritiva.
- **Decoding:** greedy em todas as abordagens. Relatar efeitos em pontos percentuais, não só p-valores.

## 8. Estudos de ablação

- **Primárias (3 seeds cada):** tamanho dos dados {250, 500, 1000, 2000, 4000}; rank {4, 8, 16, 64}; history window {0, 6, 12, 24 eventos}; k e seleção do few-shot; método (A6 vs A7 vs A8 vs A9).
- **Secundárias (cortar primeiro se faltar orçamento):** target modules, epochs, learning rate, loss masking, idle ratio, input format, inference quantization, tipo de teacher, base model.

## 9. Regras de execução

- **Treino:** número fixo de epochs por configuração. Sem early stopping nos splits de teste. A escolha de lr entre {5e-5, 1e-4, 2e-4, 4e-4} usa só Validation.
- **Exclusão de execuções:** só por falha técnica (OOM, crash, loss NaN). Registrar o motivo e repetir com a mesma seed. Nunca excluir por resultado ruim. Reportar quantos reruns ocorreram.
- **Recursos computacionais:** registrar o hardware utilizado em cada execução; não há orçamento de GPU predefinido como requisito do protocolo.

## 10. Reprodutibilidade e ameaças à validade

- **Registrar em toda execução:** config, seed, git hash, `world_hash`, `data_hash`, versões das bibliotecas, saídas brutas do modelo, uma linha em `results/*.csv`.
- **Rótulos circulares:** se o teacher é um conjunto fixo de regras, o modelo só pode copiá-lo. Mitigação: preferências ocultas inferíveis do histórico e outcome success no simulador.
- **Transferência da simulação para o mundo real:** rotinas sintéticas não são casas reais. Mitigação: Test D e limitações declaradas.
- **Sensibilidade ao prompt:** reportar a faixa entre variantes.
- **Variância do Colab e testes pequenos:** várias seeds e ≥ 1.000 linhas de teste.

## 11. Desvios (preencher após o congelamento)

| Data | Mudança | Motivo | Impacto |
|------|---------|--------|---------|
| 2026-11-22 | Retirar como requisitos orçamento de GPU, aprovação do orientador e checkpoint exato do Qwen 3B | Simplificação solicitada pelo pesquisador | Sem alteração nas hipóteses, métricas ou divisões de dados |

## 12. Checklist no momento do congelamento

- [ ] Modelo do morador definido (Step 2) e tamanhos de dataset decididos
- [x] Restrições S1–S3 e predicados O1–O3 definidos neste protocolo
- [ ] Prompts few-shot escolhidos em Validation
- [ ] Eval harness testado nas abordagens A0 e A1
- [ ] `PROTOCOL.md` e `README.md` consistentes entre si
- [x] Congelamento por decisão do pesquisador; aprovação do orientador não é requisito operacional

## 13. Registro do congelamento

- Congelamento antecipado por decisão do pesquisador em 2026-11-22, antes da conclusão da lista de verificação.
- Permanece pendente a verificação completa do avaliador. O checkpoint do Qwen 3B, um orçamento de GPU e a aprovação do orientador não são pré-requisitos do protocolo.
- O congelamento não equivale a validação científica nem aprovação institucional. Mudanças futuras devem ser documentadas como desvios, preservando esta versão.
