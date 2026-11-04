# PROTOCOL: Smart House SLM

**Versão:** v0 (rascunho) · **Data:** 2026-11-04 · **Status:** aguardando revisão do orientador
**Aluno:** Murilo Santos · **Orientador:** Dr. Paulo H. L. Rettore (UFLA) · **Projeto:** SiR.AI (PIVIC)
**Tag de congelamento:** `protocol-v1` (ainda não criada)

## 1. Regras de congelamento

- Este documento fixa perguntas, hipóteses, arms, dados, métricas e análises **antes** de qualquer avaliação nos splits de teste.
- Ele é congelado com a tag Git `protocol-v1` antes da primeira avaliação em Test A, B ou C. Depois disso, o texto não é editado: toda mudança vai na seção 11 (Desvios), com data, motivo e impacto.
- Os splits de teste nunca são usados para tuning, escolha de prompt, escolha de hyperparameters ou early stopping.

## 2. Perguntas e hipóteses

| ID | Hipótese (fixada antes de rodar) | Regra de decisão |
|----|----------------------------------|------------------|
| H1 | QLoRA SFT (A6) supera o melhor arm few-shot (A4/A5) em pelo menos 5 pontos de exact action match no **Test B** | CI de 95% (paired bootstrap) da diferença exclui 0 e a média é ≥ 5 pontos |
| H2 | A6 fica a até 3 pontos da rule policy (A1) no **Test A** | Limite inferior do CI de (A6 − A1) > −3 pontos (non-inferiority) |
| H3 | A accuracy satura por volta de 2.000 exemplos; rank acima de 16 acrescenta menos de 1 ponto | Descritiva: learning curve e ablation de rank (3 seeds); sem teste de hipótese |
| H4 | Com 20% das leituras ausentes (**Test C**), A6 perde menos pontos que o melhor arm few-shot | Comparação pareada da queda (Test B → Test C) |

Um resultado negativo (hipótese rejeitada) é relatado com o mesmo detalhe que um positivo.

## 3. Mundo simulado (versão fixa)

- **Arquivo:** `engine/house_v1.yaml`, `world_version = 1`, `world_hash = 052f8bd9f28c`. Qualquer mudança de mundo gera nova versão e novo dataset.
- **Estrutura:** 4 rooms, 7 devices, 7 sensors, tick de 10 minutos (144 ticks/dia).
- **Action space:** `light`/`switch` on/off, `lock` locked/unlocked, `climate.set_temperature` de 16 a 26 °C em passos de 0,5 (26 ações legais a partir do estado inicial).
- **Observação** (o que o modelo vê): leituras dos sensores com ruído de temperatura σ = 0,3 °C, e `p_missing` ∈ {0; 0,1; 0,2; 0,3} no Test C.
- **Dinâmica:** modelo térmico de 1ª ordem (τ = 6 h, q = 5 °C/h, hysteresis ±0,5 °C). As constantes **não são calibradas**: resultados de conforto e energia servem só para comparar arms, não como valores reais.
- **Modelo do morador:** TBD (Step 2): preferências ocultas, número de residents, tamanho da history window.

## 4. Dados e splits

| Split | Conteúdo | Uso |
|-------|----------|-----|
| Train | ~20 residents simulados, TBD dias cada | treino |
| Validation | ~4 residents | **apenas** escolha de hyperparameters e de prompt |
| Test A | dias nunca vistos dos residents de Train | in-distribution |
| Test B | residents nunca vistos | generalização (alegação principal) |
| Test C | Test B com `p_missing` 10/20/30% e layout alterado | resilience |
| Test D (opcional) | dados reais públicos, só next-room | validade externa |

- Fazer split por resident ou por dia, nunca por linha. Fazer dedup. Guardar o hash de cada split (`data_hash`).
- Train mantém todas as linhas com ação e 5% das linhas idle (seed fixa).
- O conjunto de avaliação **primário** usa a mesma regra de rebalanceamento (todas as linhas com ação + 5% idle, seed fixa). Reportamos também a visão `non-idle` e a distribuição natural.
- Mínimo de 1.000 linhas por split de teste.

## 5. Arms

Base models: `Llama-3.2-3B-Instruct` (principal), `Qwen3-4B-Instruct-2507` (replicação).

| Arm | Descrição | Settings fixos |
|-----|-----------|----------------|
| A0 | Sempre "do nothing" | sanity floor |
| A1 | Rule policy (teacher) | reference ceiling |
| A2 | ML clássico (gradient boosting ou GRU) | mesmas features |
| A3 | Zero-shot | temperature 0, `max_new_tokens = 48` |
| A4 | Few-shot | k ∈ {1, 5, 10, 20}; pool = só Train; random (3 seeds) vs nearest-neighbour |
| A5 | Few-shot + history retrieval | histórico do morador no prompt |
| A6 | **QLoRA SFT (principal)** | NF4, r = 16, alpha = 16, dropout 0, all linear layers, lr 2e-4, 1 epoch, effective batch 16, AdamW 8-bit, loss só na resposta |
| A7 | DoRA / rsLoRA | A6 com `use_dora=True`; rsLoRA com r = 64 |
| A8 | (IA)3 / prompt tuning | lr ≈ 3e-3 / ≈ 3e-2; 20 virtual tokens |
| A9 | Full fine-tuning de ~0,5B | lr ≈ 1e-5, optimizer de 8 bits |
| A10 | A6 + DPO | beta 0,1; rejected = ações inseguras ou erradas geradas pelo simulador |

Todos os arms usam o mesmo system prompt, parser, decoding (greedy), pool de treino e splits. O prompt few-shot conta nos tokens reportados. Pelo menos 3 variantes de prompt são testadas; a melhor é escolhida **só em Validation**.

## 6. Métricas

- **Primária:** exact action match (o conjunto de ações previsto é igual ao conjunto correto).
- **Secundárias:** next-room accuracy, macro-F1 por `dispositivo=valor`, format validity, safety violation rate, outcome success (closed-loop), resilience (queda com `p_missing`), efficiency (p50/p95 latency em T4 e CPU, peak VRAM, trainable params, minutos de treino, tokens por decisão).
- **Safety violations (propostas, a confirmar com o orientador):** S1 `lock.front_door` destrancada com o morador fora ou dormindo; S2 `switch.coffee` ligada com o morador fora ou dormindo; S3 setpoint > 24 °C com o morador fora ou dormindo. Saída ilegal ou não interpretável conta como erro de formato, não como violação de safety.
- **Outcome success (closed-loop, por dia simulado):** O1 temperatura a ±1 °C da preferência com o morador em casa; O2 luzes acesas só em room ocupado e escuro; O3 porta trancada com o morador fora ou dormindo. Os limiares finais dependem do Step 2 (TBD).

## 7. Plano de análise estatística

- **Seeds:** 0, 1, 2 (5 para a comparação principal, se houver tempo). Reportar média ± desvio entre seeds.
- **Inferência:** CI de 95% por paired bootstrap (10.000 resamples) nos mesmos itens de teste.
- **Família primária:** H1, H2 e H4, com Holm correction. H3 é descritiva.
- **Decoding:** greedy em todos os arms. Reportar efeitos em pontos percentuais, não só p-valores.

## 8. Ablations

- **Primárias (3 seeds cada):** tamanho dos dados {250, 500, 1000, 2000, 4000}; rank {4, 8, 16, 64}; history window {0, 6, 12, 24 eventos}; k e seleção do few-shot; método (A6 vs A7 vs A8 vs A9).
- **Secundárias (cortar primeiro se faltar budget):** target modules, epochs, learning rate, loss masking, idle ratio, input format, inference quantization, tipo de teacher, base model.

## 9. Regras de execução

- **Treino:** número fixo de epochs por configuração. Sem early stopping nos splits de teste. A escolha de lr entre {5e-5, 1e-4, 2e-4, 4e-4} usa só Validation.
- **Exclusão de runs:** só por falha técnica (OOM, crash, loss NaN). Registrar o motivo e repetir com a mesma seed. Nunca excluir por resultado ruim. Reportar quantos reruns ocorreram.
- **Budget:** Colab T4; limite total de GPU-hours TBD. Se acabar, cortar primeiro as ablations secundárias, depois reduzir seeds de 3 para 2 nas primárias (declarando isso).

## 10. Reprodutibilidade e ameaças à validade

- **Registrar em todo run:** config, seed, git hash, `world_hash`, `data_hash`, versões das bibliotecas, saídas brutas do modelo, uma linha em `results/*.csv`.
- **Labels circulares:** se o teacher é um conjunto fixo de regras, o modelo só pode copiá-lo. Mitigação: preferências ocultas inferíveis do histórico e outcome success no simulador.
- **Sim-to-real:** rotinas sintéticas não são casas reais. Mitigação: Test D e limitações declaradas.
- **Sensibilidade ao prompt:** reportar a faixa entre variantes.
- **Variância do Colab e testes pequenos:** várias seeds e ≥ 1.000 linhas de teste.

## 11. Desvios (preencher após o congelamento)

| Data | Mudança | Motivo | Impacto |
|------|---------|--------|---------|
| | | | |

## 12. Checklist antes de congelar

- [ ] Modelo do morador definido (Step 2) e tamanhos de dataset decididos
- [ ] Safety constraints S1–S3 e predicados O1–O3 confirmados com o orientador
- [ ] Prompts few-shot escolhidos em Validation
- [ ] Eval harness testado nos arms A0 e A1
- [ ] `PROTOCOL.md` e `README.md` consistentes entre si
- [ ] Aprovação do orientador registrada, depois `git tag protocol-v1`
