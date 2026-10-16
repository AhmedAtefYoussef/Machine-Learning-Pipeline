# Markdown cells that refer to removed/merged outputs after W3a (aaf9f9b)

- P0 setup md: "core algorithms are printed (show_source)" -> only 4 now (GD+mse_grad, lift_weights, soft_threshold+cd_sweep, logloss_grad+fit_logistic); Target not printed.
- P1 J representation: hour-encoding table gone; p1.hour_encoding_ablation.* placeholders remain in text without a table (keep numbers inline in prose).
- P1 Live check / J learning rate: one plot (loss curve) + sweep table now.
- P1 J target transform: "the table below" ok; exponent plot gone.
- P1 J bonus: bonus plot, asym source and separate table gone; one 3-row table (MSE, k=1 control, asymmetric) follows "A control for the bonus".
- P1 Outcome: train/val R2 and loss curve now in the correctness table; residual plot kept.
- P2 J what we expanded / J degree: ablation + power-column sweep + degree sweep are ONE combined table after the degree cell; degree plot gone.
- P3 J what we varied: degree-axis plot gone. J chronological cut: estimates plot gone, table kept. J which estimate to trust: chrono_detail and noise_floor tables gone. J exponent check: plot gone, table kept. Outcome: hard-typed diagnostics -> placeholders; noise_floor not shown.
- P4 J search ranges: design table and ridge source gone; unregularised row in methods table. J fair comparison: enet ratio table gone. J choice of lambda: cv-vs-val plot, ridge path, entry table gone; lasso path kept. J verdict rule: stability plot gone. J final model: one stage-B table, recommended model in caption; stage-B curves and rich-level check gone. Outcome mentions rich-level check.
- P5 J label rule: label-variants plot gone. J operating threshold: threshold table, calibration table+plot gone (give calibration max gap in Outcome via placeholder if key exists). J classifier regularisation: design table gone.
- P6 intro: summary/submission tables replaced by printed lines + asserts. "The pipeline at a glance" md is now the last cell with no code after it.
