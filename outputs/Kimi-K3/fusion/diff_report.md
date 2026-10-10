# 融合差异报告（机械融合产出，供复核裁决）

- IR 节点（含追加）：69；状态分布：{"mismatch": 42, "verified": 21, "llm_only": 6}
- 实测记录：12086 条；已对齐 8819 条；未覆盖模块路径 0 个

## 场景符号绑定表
- decode：B=2、Dk=32、Ec=256、Ei=128、El=256、H=512、Hq=8、I=1024、Ic=2048、Is=256、Isc=512、Kv=32、Ne=2、P=256、Qr=64、S2=1、T=2
- prefill_text：B=2、Dk=32、Ec=256、Ei=128、El=256、H=512、Hq=8、I=1024、Ic=2048、Is=256、Isc=512、Kv=32、Ne=32、P=256、Qr=64、S=16、S2=16、T=32
- prefill_vision：B=2、Cv=128、Ec=256、Ei=128、El=256、H=512、I=1024、Ic=2048、Is=256、Isc=512、Kv=32、Lv=72、Ne=43、Nv=17、Nv0=16、P=128、Qkv=128、Qr=64、S2=27、T=54

## 不一致节点（须复核）（42）
- `vision_tower`：[{"scenario": "prefill_vision", "notes": ["__getitem___304（full 对齐）：维度数不同：逻辑 ['Lv', 3, 14, 14] vs 实测 [72, 128]", "view_305（full 对齐）：维度数不同：逻辑 ['Lv', 3, 14, 14] vs 实测 [64, 128]", "permute_307（full 对齐）：维度数不同：逻辑 ['Lv', 3, 14, 14] vs 实测 [1, 4, 2, 4, 2, 128]", "contiguous_309（full 对齐）：维度数不同：逻辑 ['Lv', 3, 14, 14] vs 实测 [1, 4, 4, 2, 2, 128]", "mean_310（full 对齐）：维度数不同：逻辑 ['Lv', 3, 14, 14] vs 实测 [1, 4, 4, 2,
- `vision_tower.patch_embed`：[{"scenario": "prefill_vision", "notes": ["view_23（full 对齐）：第 1 维不同：逻辑 3 vs 实测 128", "view_7（full 对齐）：第 1 维不同：逻辑 3 vs 实测 128"]}]
- `vision_tower.patch_embed.pos_emb`：[{"scenario": "prefill_vision", "notes": ["flatten_27（full 对齐）：维度数不同：逻辑 ['Lv', 'Cv'] vs 实测 [8, 8, 128]", "reshape_29（full 对齐）：第 0 维符号 Lv 在本场景内绑定冲突：既有 72 又有 64（维度语义不一致）", "permute_39（full 对齐）：维度数不同：逻辑 ['Lv', 'Cv'] vs 实测 [8, 8, 128]", "unsqueeze_40（full 对齐）：维度数不同：逻辑 ['Lv', 'Cv'] vs 实测 [128, 8, 8]", "interpolate_42（full 对齐）：维度数不同：逻辑 ['Lv', 'Cv'] vs 实测 [1, 128, 8, 8]", "squeeze_43（full 对齐）：维度数不同：逻辑 ['
- `vision_tower.encoder`：[{"scenario": "prefill_vision", "notes": ["float_60（full 对齐）：维度数不同：逻辑 ['Lv', 'Cv'] vs 实测 [262144]", "to_62（full 对齐）：维度数不同：逻辑 ['Lv', 'Cv'] vs 实测 [262144]", "__mod___63（full 对齐）：维度数不同：逻辑 ['Lv', 'Cv'] vs 实测 [262144]", "__floordiv___65（full 对齐）：维度数不同：逻辑 ['Lv', 'Cv'] vs 实测 [262144]", "__getitem___68（full 对齐）：维度数不同：逻辑 ['Lv', 'Cv'] vs 实测 [8]", "float_70（full 对齐）：维度数不同：逻辑 ['Lv', 'Cv'] vs 实测 [8]", "to_71（f
- `vision_tower.encoder.{j}`：[{"scenario": "prefill_vision", "notes": ["view_122（full 对齐）：第 1 维符号 Cv 在本场景内绑定冲突：既有 128 又有 384（维度语义不一致）", "unbind_124（full 对齐）：维度数不同：逻辑 ['Lv', 'Cv'] vs 实测 [72, 3, 4, 32]", "unsqueeze_125（full 对齐）：第 1 维符号 Cv 在本场景内绑定冲突：既有 128 又有 16（维度语义不一致）", "float_127（full 对齐）：维度数不同：逻辑 ['Lv', 'Cv'] vs 实测 [72, 4, 32]", "view_128（full 对齐）：维度数不同：逻辑 ['Lv', 'Cv'] vs 实测 [72, 4, 32]", "view_as_complex_130（full 对齐）：维度数不同
- `vision_tower.encoder.{j}.mlp.fc1`：[{"scenario": "prefill_vision", "notes": ["linear_202（full 对齐）：第 1 维不同：逻辑 4096 vs 实测 256", "linear_294（full 对齐）：第 1 维不同：逻辑 4096 vs 实测 256", "linear_88（full 对齐）：第 1 维不同：逻辑 4096 vs 实测 256"]}]
- `mm_projector`：[{"scenario": "prefill_vision", "notes": ["view_325（full 对齐）：第 0 维符号 Nv0 在本场景内绑定冲突：既有 16 又有 1（维度语义不一致）", "concat_327（full 对齐）：维度数不同：逻辑 ['Nv0', 4, 'Cv'] vs 实测 [16, 512]", "split_341（full 对齐）：维度数不同：逻辑 ['Nv0', 4, 'Cv'] vs 实测 [17, 512]", "view_3（full 对齐）：第 0 维符号 Nv0 在本场景内绑定冲突：既有 16 又有 4（维度语义不一致）", "concat_5（full 对齐）：维度数不同：逻辑 ['Nv0', 4, 'Cv'] vs 实测 [16, 512]", "split_19（full 对齐）：维度数不同：逻辑 ['Nv0', 4, 'Cv
- `mm_projector.proj.0`：[{"scenario": "prefill_vision", "notes": ["linear_330（full 对齐）：第 1 维不同：逻辑 4096 vs 实测 512", "linear_8（full 对齐）：第 0 维符号 Nv 在本场景内绑定冲突：既有 17 又有 20（维度语义不一致）"]}]
- `mm_projector.proj.1`：[{"scenario": "prefill_vision", "notes": ["gelu_333（full 对齐）：第 1 维不同：逻辑 4096 vs 实测 512", "gelu_11（full 对齐）：第 0 维符号 Nv 在本场景内绑定冲突：既有 17 又有 20（维度语义不一致）"]}]
- `mm_projector.proj.2`：[{"scenario": "prefill_vision", "notes": ["linear_335（full 对齐）：第 1 维不同：逻辑 4096 vs 实测 512", "linear_13（full 对齐）：第 0 维符号 Nv 在本场景内绑定冲突：既有 17 又有 20（维度语义不一致）"]}]
- `mm_projector.post_norm`：[{"scenario": "prefill_vision", "notes": ["rms_norm_16（full 对齐）：第 0 维符号 Nv 在本场景内绑定冲突：既有 17 又有 20（维度语义不一致）"]}]
- `language_model.model.{i}`：[{"scenario": "decode", "notes": ["unsqueeze_29（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [2, 512]", "cat_31（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 0（维度语义不一致）", "unsqueeze_213（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [2, 512]", "float_216（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 2（维度语义不一致）", "pow_217（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 2（维度语义不一致）", "mean_219（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 2（维度
- `language_model.model.{i}.input_layernorm`：[{"scenario": "decode", "notes": ["pow_35（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "mean_37（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "__add___39（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 1（维度语义不一致）", "rsqrt_41（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 1（维度语义不一致）", "__mul___42（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "to_43（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "__mul___44（full 对齐）
- `language_model.model.{i}.post_attention_layernorm`：[{"scenario": "decode", "notes": ["pow_247（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "mean_249（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "__add___251（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 1（维度语义不一致）", "rsqrt_253（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 1（维度语义不一致）", "__mul___254（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "to_255（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "__mul___256（f
- `language_model.model.{i}.self_attn`：[{"scenario": "decode", "notes": ["reshape_99（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 256（维度语义不一致）", "permute_100（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [2, 1, 8, 32]", "reshape_102（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [2, 1, 8, 32]", "float_106（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 8（维度语义不一致）", "reshape_108（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 256（维度语义不一致）", "permute_109（full 对齐）：维度数不同：逻辑
- `language_model.model.{i}.self_attn.q_proj`：[{"scenario": "prefill_vision", "notes": ["linear_485（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_1933（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_3581（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）"]}]
- `language_model.model.{i}.self_attn.k_proj`：[{"scenario": "prefill_vision", "notes": ["linear_488（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_1936（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_3584（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）"]}]
- `language_model.model.{i}.self_attn.v_proj`：[{"scenario": "prefill_vision", "notes": ["linear_491（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_1939（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_3587（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）"]}]
- `language_model.model.{i}.self_attn.q_conv1d`：[{"scenario": "decode", "notes": ["cat_59（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "conv1d_60（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "transpose_63（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "__getitem___65（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "contiguous_66（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "cat_350（full 对齐）：第 1 维符号 S2 在本场景内绑定冲
- `language_model.model.{i}.self_attn.k_conv1d`：[{"scenario": "decode", "notes": ["cat_71（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "conv1d_72（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "transpose_75（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "__getitem___77（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "contiguous_78（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "cat_362（full 对齐）：第 1 维符号 S2 在本场景内绑定冲
- `language_model.model.{i}.self_attn.v_conv1d`：[{"scenario": "decode", "notes": ["cat_83（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "conv1d_84（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "transpose_87（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "__getitem___89（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "contiguous_90（full 对齐）：第 1 维符号 S2 在本场景内绑定冲突：既有 1 又有 256（维度语义不一致）", "cat_374（full 对齐）：第 1 维符号 S2 在本场景内绑定冲
- `language_model.model.{i}.self_attn.f_a_proj`：[{"scenario": "prefill_vision", "notes": ["linear_647（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_2095（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_3743（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）"]}]
- `language_model.model.{i}.self_attn.f_b_proj`：[{"scenario": "prefill_vision", "notes": ["linear_650（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_2098（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_3746（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）"]}]
- `language_model.model.{i}.self_attn.b_proj`：[{"scenario": "prefill_vision", "notes": ["linear_658（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_2106（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_3754（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）"]}]
- `language_model.model.{i}.self_attn.o_norm`：[{"scenario": "decode", "notes": ["pow_188（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "mean_190（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "__add___192（full 对齐）：第 3 维符号 Dk 在本场景内绑定冲突：既有 32 又有 1（维度语义不一致）", "rsqrt_194（full 对齐）：第 3 维符号 Dk 在本场景内绑定冲突：既有 32 又有 1（维度语义不一致）", "__mul___195（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "float_196（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'Hq', 'Dk'] vs 实测 [32]", "_
- `language_model.model.{i}.self_attn.g_proj`：[{"scenario": "prefill_vision", "notes": ["linear_1743（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_3191（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_4839（full 对齐）：第 0 维符号 B 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）"]}]
- `language_model.model.{i}.self_attn.o_proj`：[{"scenario": "decode", "notes": ["linear_1331（full 对齐）：第 2 维符号 P 在本场景内绑定冲突：既有 256 又有 128（维度语义不一致）", "linear_80（full 对齐）：第 2 维符号 P 在本场景内绑定冲突：既有 256 又有 128（维度语义不一致）"]}, {"scenario": "prefill_text", "notes": ["linear_2422（full 对齐）：第 2 维符号 P 在本场景内绑定冲突：既有 256 又有 128（维度语义不一致）", "linear_92（full 对齐）：第 2 维符号 P 在本场景内绑定冲突：既有 256 又有 128（维度语义不一致）", "linear_140（full 对齐）：第 2 维符号 P 在本场景内绑定冲突：既有 256 又有 128（维度语义不一
- `language_model.model.{i}.self_attn.q_a_layernorm`：[{"scenario": "decode", "notes": ["pow_1257（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "mean_1259（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "__add___1261（full 对齐）：第 2 维符号 Qr 在本场景内绑定冲突：既有 64 又有 1（维度语义不一致）", "rsqrt_1263（full 对齐）：第 2 维符号 Qr 在本场景内绑定冲突：既有 64 又有 1（维度语义不一致）", "__mul___1264（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "to_1265（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "__mul__
- `language_model.model.{i}.self_attn.kv_a_layernorm`：[{"scenario": "decode", "notes": ["pow_1281（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "mean_1283（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "__add___1285（full 对齐）：第 2 维符号 Kv 在本场景内绑定冲突：既有 32 又有 1（维度语义不一致）", "rsqrt_1287（full 对齐）：第 2 维符号 Kv 在本场景内绑定冲突：既有 32 又有 1（维度语义不一致）", "__mul___1288（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "to_1289（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "__mul__
- `language_model.model.{i}.block_sparse_moe`：[{"scenario": "decode", "notes": ["scatter__579（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [2, 4]", "sum_582（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [2, 4]", "view_583（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [2, 2]", "argsort_585（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [4]", "__floordiv___586（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [4]", "__getitem___588（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 
- `language_model.model.{i}.block_sparse_moe.gate`：[{"scenario": "decode", "notes": ["type_554（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [2, 512]", "type_555（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [4, 512]", "linear_557（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [2, 512]", "sigmoid_558（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [2, 4]", "view_559（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测 [2, 4]", "unsqueeze_561（full 对齐）：维度数不同：逻辑 ['B', 'S2', 'H'] vs 实测
- `language_model.model.{i}.block_sparse_moe.routed_expert_norm`：[{"scenario": "decode", "notes": ["pow_696（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "mean_698（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "__add___700（full 对齐）：第 1 维符号 El 在本场景内绑定冲突：既有 256 又有 1（维度语义不一致）", "rsqrt_702（full 对齐）：第 1 维符号 El 在本场景内绑定冲突：既有 256 又有 1（维度语义不一致）", "__mul___703（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "to_704（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "__mul___705
- `language_model.model.{i}.block_sparse_moe.shared_experts`：[{"scenario": "decode", "notes": ["cat_719（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 256（维度语义不一致）", "cat_1178（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 256（维度语义不一致）", "cat_1519（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 256（维度语义不一致）"]}, {"scenario": "prefill_text", "notes": ["cat_1450（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 256（维度语义不一致）", "cat_2257（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 256（维度语义不一致）", "cat_2
- `language_model.model.0.mlp`：[{"scenario": "decode", "notes": ["cat_265（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 1024（维度语义不一致）"]}, {"scenario": "prefill_text", "notes": ["cat_648（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 1024（维度语义不一致）", "cat_611（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 1024（维度语义不一致）", "cat_7（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 1024（维度语义不一致）"]}, {"scenario": "prefill_vision", "notes": ["cat_1835（full 对齐）：第 2 维符号 H
- `language_model.model.norm`：[{"scenario": "decode", "notes": ["pow_1580（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "mean_1582（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "__add___1584（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 1（维度语义不一致）", "rsqrt_1586（full 对齐）：第 2 维符号 H 在本场景内绑定冲突：既有 512 又有 1（维度语义不一致）", "__mul___1587（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "to_1588（full 对齐）：dtype 不同：逻辑 bfloat16 vs 实测 float32", "__mul__
- `language_model.model.{i}.block_sparse_moe.{j}`：[{"scenario": "decode", "notes": ["cat_599（full 对齐）：第 1 维符号 El 在本场景内绑定冲突：既有 256 又有 128（维度语义不一致）", "cat_630（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "cat_661（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "cat_1058（full 对齐）：第 1 维符号 El 在本场景内绑定冲突：既有 256 又有 128（维度语义不一致）", "cat_1089（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "cat_1120（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一
- `language_model.model.{i}.block_sparse_moe.{j}.w1`：[{"scenario": "decode", "notes": ["linear_625（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_656（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_1084（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_1115（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）"]}, {"scenario": "prefill_text", "notes": ["linear_2（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 32 又有 5（维度语义不一致）", "linear_6
- `language_model.model.{i}.block_sparse_moe.{j}.w2`：[{"scenario": "decode", "notes": ["linear_651（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_682（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_1110（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_1141（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）"]}, {"scenario": "prefill_text", "notes": ["linear_28（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 32 又有 5（维度语义不一致）", "linear_
- `language_model.model.{i}.block_sparse_moe.{j}.w3`：[{"scenario": "decode", "notes": ["linear_628（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_659（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_1087（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）", "linear_1118（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 2 又有 1（维度语义不一致）"]}, {"scenario": "prefill_text", "notes": ["linear_5（full 对齐）：第 0 维符号 Ne 在本场景内绑定冲突：既有 32 又有 5（维度语义不一致）", "linear_6
- `language_model.model.0.mlp.act_fn`：[{"scenario": "decode", "notes": ["to_268（full 对齐）：第 2 维符号 Ic 在本场景内绑定冲突：既有 2048 又有 1024（维度语义不一致）", "to_270（full 对齐）：第 2 维符号 Ic 在本场景内绑定冲突：既有 2048 又有 1024（维度语义不一致）", "__truediv___271（full 对齐）：第 2 维符号 Ic 在本场景内绑定冲突：既有 2048 又有 1024（维度语义不一致）", "tanh_273（full 对齐）：第 2 维符号 Ic 在本场景内绑定冲突：既有 2048 又有 1024（维度语义不一致）", "__rmul___274（full 对齐）：第 2 维符号 Ic 在本场景内绑定冲突：既有 2048 又有 1024（维度语义不一致）", "sigmoid_276（full 对齐）：第 
- `language_model.model.{i}.block_sparse_moe.{j}.act_fn`：[{"scenario": "decode", "notes": ["to_602（full 对齐）：第 1 维符号 Ec 在本场景内绑定冲突：既有 256 又有 128（维度语义不一致）", "to_604（full 对齐）：第 1 维符号 Ec 在本场景内绑定冲突：既有 256 又有 128（维度语义不一致）", "__truediv___605（full 对齐）：第 1 维符号 Ec 在本场景内绑定冲突：既有 256 又有 128（维度语义不一致）", "tanh_607（full 对齐）：第 1 维符号 Ec 在本场景内绑定冲突：既有 256 又有 128（维度语义不一致）", "__rmul___608（full 对齐）：第 1 维符号 Ec 在本场景内绑定冲突：既有 256 又有 128（维度语义不一致）", "sigmoid_610（full 对齐）：第 1 维符号 Ec 在
- `language_model.model.{i}.block_sparse_moe.shared_experts.act_fn`：[{"scenario": "decode", "notes": ["to_722（full 对齐）：第 2 维符号 Isc 在本场景内绑定冲突：既有 512 又有 256（维度语义不一致）", "to_724（full 对齐）：第 2 维符号 Isc 在本场景内绑定冲突：既有 512 又有 256（维度语义不一致）", "__truediv___725（full 对齐）：第 2 维符号 Isc 在本场景内绑定冲突：既有 512 又有 256（维度语义不一致）", "tanh_727（full 对齐）：第 2 维符号 Isc 在本场景内绑定冲突：既有 512 又有 256（维度语义不一致）", "__rmul___728（full 对齐）：第 2 维符号 Isc 在本场景内绑定冲突：既有 512 又有 256（维度语义不一致）", "sigmoid_730（full 对齐）：第 2 维符号

## 仅逻辑节点（6）
- `vision_tower.encoder.{j}.mlp`：任何场景均未测到（未执行 ≠ 不存在，页面将带标注）
- `vision_tower.encoder.rope_2d`：任何场景均未测到（未执行 ≠ 不存在，页面将带标注）
- `merge`：任何场景均未测到（未执行 ≠ 不存在，页面将带标注）
- `language_model.model.{i}.attn_res`：任何场景均未测到（未执行 ≠ 不存在，页面将带标注）
- `lm.cache`：任何场景均未测到（未执行 ≠ 不存在，页面将带标注）
- `vision_tower.encoder.{j}.mlp.activation`：任何场景均未测到（未执行 ≠ 不存在，页面将带标注）

## 仅实测节点（2a 未覆盖）（0）

## 已验证节点（21）
- `vision_tower.patch_embed.proj`：2 条实测记录全部一致（场景：['prefill_vision']）
- `vision_tower.encoder.{j}.norm0`：3 条实测记录全部一致（场景：['prefill_vision']）
- `vision_tower.encoder.{j}.wqkv`：3 条实测记录全部一致（场景：['prefill_vision']）
- `vision_tower.encoder.{j}.wo`：3 条实测记录全部一致（场景：['prefill_vision']）
- `vision_tower.encoder.{j}.mlp.fc0`：3 条实测记录全部一致（场景：['prefill_vision']）
- `vision_tower.encoder.{j}.norm1`：3 条实测记录全部一致（场景：['prefill_vision']）
- `vision_tower.encoder.final_layernorm`：1 条实测记录全部一致（场景：['prefill_vision']）
- `language_model.model.embed_tokens`：1 条实测记录全部一致（场景：['prefill_text']）
- `language_model.model.{i}.self_attn.q_a_proj`：6 条实测记录全部一致（场景：['decode', 'prefill_text', 'prefill_vision']）
- `language_model.model.{i}.self_attn.q_b_proj`：6 条实测记录全部一致（场景：['decode', 'prefill_text', 'prefill_vision']）
- `language_model.model.{i}.self_attn.kv_a_proj_with_mqa`：6 条实测记录全部一致（场景：['decode', 'prefill_text', 'prefill_vision']）
- `language_model.model.{i}.self_attn.kv_b_proj`：6 条实测记录全部一致（场景：['decode', 'prefill_text', 'prefill_vision']）
- `language_model.model.{i}.block_sparse_moe.routed_expert_down_proj`：11 条实测记录全部一致（场景：['decode', 'prefill_text', 'prefill_vision']）
- `language_model.model.{i}.block_sparse_moe.routed_expert_up_proj`：11 条实测记录全部一致（场景：['decode', 'prefill_text', 'prefill_vision']）
- `language_model.model.{i}.block_sparse_moe.shared_experts.gate_proj`：11 条实测记录全部一致（场景：['decode', 'prefill_text', 'prefill_vision']）
- `language_model.model.{i}.block_sparse_moe.shared_experts.up_proj`：11 条实测记录全部一致（场景：['decode', 'prefill_text', 'prefill_vision']）
- `language_model.model.{i}.block_sparse_moe.shared_experts.down_proj`：11 条实测记录全部一致（场景：['decode', 'prefill_text', 'prefill_vision']）
- `language_model.model.0.mlp.gate_proj`：5 条实测记录全部一致（场景：['decode', 'prefill_text', 'prefill_vision']）
- `language_model.model.0.mlp.up_proj`：5 条实测记录全部一致（场景：['decode', 'prefill_text', 'prefill_vision']）
- `language_model.model.0.mlp.down_proj`：5 条实测记录全部一致（场景：['decode', 'prefill_text', 'prefill_vision']）
- `language_model.lm_head`：4 条实测记录全部一致（场景：['decode', 'prefill_text', 'prefill_vision']）

## 复核指引
- mismatch：先核对 evidence 的行号是否真的支持逻辑侧；再看对齐是否正确（路径模板 vs 实测树路径）；最后考虑替换件是否改变了行为。
- 修正逻辑侧时必须同步补/改 evidence；无法裁决的把 status 改为 conflict 并写 open_questions。
- capture_only：属于逻辑分析的覆盖缺口，补分析（或确认辅助路径后在 concerns 说明）。

## 复核记录（S3 人工裁决 · attempt 3）

> 断点续作说明：本报告上文的机械差异清单由 fuse.py 重新生成（attempt 3 重跑，机械草稿
> 曾被前序代理中断在融合复核阶段、另一代理基于未裁决草稿做过阶段五/六，定稿后两者均按
> 本复核结果重做）。复核按 prompts/fusion-adjudication.md 三问归因执行：
> ①逻辑侧证据是否成立 ②对齐是否正确（路径模板/记录选择）③替换件是否改变行为。

### 归因结论总览

- 42 个 mismatch **全部完成三问归因**，未发现无法裁决项，全部转 verified；无 conflict。
- 归因分布（三问口径）：
  1. **逻辑侧字面量未随缩水实例化**（逻辑修正+同步 evidence）：fc1 4096→Iv、proj.0/1/2
     4096→Hp —— 2 处根因，3 个节点。
  2. **逻辑侧遗漏源码分支**：KDA unpad 展平域（q/k/v_proj、f_a/f_b/b_proj、conv1d、
     o_norm、g_proj、o_proj 的 Bu/Su 域符号）、Rope2D 硬编码 512×512 预计算表（encoder 节点）。
  3. **机械融合口径现象（逻辑与实测均正确）**：场景级全局符号绑定使不同张量的同下标维
     并入同符号（H/P/Cv/Lv/Nv0 等绑定冲突）、双层型共享模板（o_proj/g_proj 的 P vs Hq·Vd）、
     模板 I/O 域 vs 记录内部域（塔/patch_embed/pos_emb/层/attn_res/moe 簿记）、
     fp32 内部算子 vs bf16 I/O（KimiRMSNorm 家族、SituAndMul）、运行时动态值（Ne、Nv0）。
  4. **替换件行为（如实声明，非毛坯本征）**：conv1d 域布局与 cache (Bu,P,K-1)（R2 shim）、
     o_norm 的 FusedRMSNormGated shim、MLA 的 FA2 等价函数域内 fp32 算子（R3）。
- llm_only 6 个的处置：`language_model.model.{i}.attn_res` 升级 verified（记录自层节点回填，
  普通方法共享 tree_path 无独立 scope）；`vision_tower.encoder.rope_2d` 修正 logical
  （64→Dr）后保留 llm_only（普通函数无自有 scope，encoder 记录为旁证）；
  `vision_tower.encoder.{j}.mlp`、`.activation`、`merge`、`lm.cache` 保留 llm_only
  （mlp/activation：gelu 算子未被 torchvista 登记，仅有源码与形状链旁证；merge：根域记录
  无法逐条区分归因；lm.cache：非 Module 状态容器无前向算子）。

### 逐节点归因（42 行）

| 节点 | 归因 |
|---|---|
| `vision_tower` | 记录=tpool_patch_merger 时间池化/空间合并链（塔级普通函数，:621-646）；塔输入域由 patch_embed.proj 旁证 → verified |
| `vision_tower.patch_embed` | 记录=proj 输出 view(L,-1)（:335）；输入域由 proj 节点旁证 → verified |
| `vision_tower.patch_embed.pos_emb` | 逐图直取/双线性插值/时间 repeat/cat/add 全链对上（:247-283）；逐图中间量与 Lv 并符号属全局绑定 → verified |
| `vision_tower.encoder` | rope 预计算表 262144=硬编码 512×512（:579-582，毛坯本征）+逐图切片+cu_seqlens 累积（:603-610）→ verified |
| `vision_tower.encoder.{j}` | qkv 拆分/apply_rope/eager 注意力/残差逐条对上（:461-543,:102-139,:172-193）；Cv 并 384 属全局绑定 → verified |
| `vision_tower.encoder.{j}.mlp.fc1` | 逻辑侧照抄全量字面量 4096，缩水 Iv=256；改符号 + 同步 evidence（:437-458）→ verified |
| `mm_projector` | list 逐图 view/concat/split 链对上（:783-815）；Nv0 多值=逐图/跨作业组合差异 → verified |
| `mm_projector.proj.0` | 逻辑侧全量字面量 4096；实际 Hp=4×mm_hidden=512（:789-795）；改符号 Hp → verified |
| `mm_projector.proj.1` | 同 proj.0；GELU 在 Hp=512 域 → verified |
| `mm_projector.proj.2` | proj 末层 Linear(Hp→text_hidden)，缩水下两维同值 512 → verified |
| `mm_projector.post_norm` | post_norm 在 text_hidden 域（缩水 512）；Nv 跨作业 17/20=输入组合差异 → verified |
| `language_model.model.{i}` | 散件记录=层内 attn_res 机制（:973-1046,:1075-1088,:1188-1192），decode 链逐算子对上；拼接前后维并符号属全局绑定 → verified |
| `language_model.model.{i}.input_layernorm` | KimiRMSNorm fp32 内部（:226-236）：I/O bf16 与中间 fp32 并存，机械按中间 dtype 报差异 → verified |
| `language_model.model.{i}.post_attention_layernorm` | 同 input_layernorm（fp32 内部口径）→ verified |
| `language_model.model.{i}.self_attn` | 双层型（KDA/MLA）共享模板的全部内部算子；逐条归位至 :405-473 与 :543-664，子节点分治承载 → verified |
| `language_model.model.{i}.self_attn.q_proj` | B 2/1=含图作业双支路（文本 (2,27)/视觉 unpad (1,39)，:566-574）；改展平符号 Bu/Su → verified |
| `language_model.model.{i}.self_attn.k_proj` | 同 q_proj（双支路展平域）→ verified |
| `language_model.model.{i}.self_attn.v_proj` | 同 q_proj（双支路展平域）→ verified |
| `language_model.model.{i}.self_attn.q_conv1d` | 卷积域 (Bu,P,K') 中间量 dim1=通道维 P 与序列维并符号（:583-600）；cache (Bu,P,K-1) 为 shim 定义布局（R2）→ verified |
| `language_model.model.{i}.self_attn.k_conv1d` | 同 q_conv1d（卷积域布局差异）→ verified |
| `language_model.model.{i}.self_attn.v_conv1d` | 同 q_conv1d（卷积域布局差异）→ verified |
| `language_model.model.{i}.self_attn.f_a_proj` | f_a=Linear(hidden,head_dim)（:523）；双支路展平域，改 Bu/Su → verified |
| `language_model.model.{i}.self_attn.f_b_proj` | f_b=Linear(head_dim,projection_size)（:524）：输入域 Dk、输出 P；改 Bu/Su → verified |
| `language_model.model.{i}.self_attn.b_proj` | b_proj=Linear(hidden,num_heads)（:529）输出 beta 头数 Hk=8；双支路展平域 → verified |
| `language_model.model.{i}.self_attn.o_norm` | FusedRMSNormGated shim（R2）：I/O bf16 + 内部 fp32 + weight/keepdim 广播（:539-541,:656）；改 (Bu,Su,Hk,Dk) → verified |
| `language_model.model.{i}.self_attn.g_proj` | 双层型：KDA 输出 P=256（:534）/MLA 输出 Hq·Vd=128（:401），逐条归位；改 Bu/Su/Gout → verified |
| `language_model.model.{i}.self_attn.o_proj` | 双层型：KDA 输入 P=256（:541）/MLA 输入 Hq·Vd=128（:390），逐条归位；改 (Bx,Sx,Oin) → verified |
| `language_model.model.{i}.self_attn.q_a_layernorm` | KimiRMSNorm fp32 内部口径（Qr=64 缩水）→ verified |
| `language_model.model.{i}.self_attn.kv_a_layernorm` | KimiRMSNorm fp32 内部口径（Kv=32 缩水）→ verified |
| `language_model.model.{i}.block_sparse_moe` | moe_infer 簿记（argsort/scatter/切片/cat，:841-877）在展平域 (T,El)；专家/路由由子节点承载 → verified |
| `language_model.model.{i}.block_sparse_moe.gate` | 路由链 fp32 logits/sigmoid/top-2/renorm 对上（:703-746）；grouped 分支未激活与逻辑一致 → verified |
| `language_model.model.{i}.block_sparse_moe.routed_expert_norm` | KimiRMSNorm fp32 内部，展平域 (T,El=256) → verified |
| `language_model.model.{i}.block_sparse_moe.shared_experts` | gate/up cat (·,Is=256)→(·,Isc=512) 对上；cat 前后维并符号属全局绑定 → verified |
| `language_model.model.0.mlp` | dense cat (·,I=1024)→(·,Ic=2048) 对上；并符号属全局绑定 → verified |
| `language_model.model.norm` | KimiRMSNorm fp32 内部口径 → verified |
| `language_model.model.{i}.block_sparse_moe.{j}` | 单专家 w1/w3→cat→situ→w2 对上（:242-260）；Ne 多值=动态分派本征（跨作业/跨实例）→ verified |
| `language_model.model.{i}.block_sparse_moe.{j}.w1` | w1 (Ne,El)→(Ne,Ei)；Ne 动态 → verified |
| `language_model.model.{i}.block_sparse_moe.{j}.w2` | w2 (Ne,Ei)→(Ne,El)；Ne 动态 → verified |
| `language_model.model.{i}.block_sparse_moe.{j}.w3` | w3 (Ne,El)→(Ne,Ei)；Ne 动态 → verified |
| `language_model.model.0.mlp.act_fn` | SituAndMul 切半语义：入 (·,2I) 出 (·,I)（:64-82）；前后维并符号 Ic 属机械绑定 → verified |
| `language_model.model.{i}.block_sparse_moe.{j}.act_fn` | SituAndMul 切半（Ec/Ei）→ verified |
| `language_model.model.{i}.block_sparse_moe.shared_experts.act_fn` | SituAndMul 切半（Isc/Is）→ verified |

### 契约层补全（定稿内容）

- `contract.replacements`：4 项（R1 einops rearrange 子集 / R2 fla 六组件 / R3 FA2 CPU 等价 /
  P transformers 5.17.0 兼容补丁），各含 what/why/equiv/evidence/shim_class。
  spec.shims=[] 与之不矛盾：本模型替换件均为导入级/注册级，不经 shims[] 的类挂树机制
  （见 capture/spec.json notes）。
- `contract.coverage.captured`：20 个作业逐项列出（3 宏 + 17 micro），每项注明覆盖面；
  folded 更新为代表层覆盖口径。
- `model.config_snapshot`：改为实际实例化（hooks.py TEXT_CFG/VISION_CFG），
  原全量快照保留于「config_json_全量对照」子键。
- `model.scale_note`：重写为「分支保真 + 逐项缩水数值对照」；修正 llm-analysis/notes.md
  初稿的 proj 中间维「4096→128」笔误（实际=4×mm_hidden，缩水 512）——修正以 notes.md
  追加条目方式留痕（原文不回改）。
- 定稿状态分布：{'verified': 64, 'llm_only': 5}（verified 64 / llm_only 5 /
  mismatch 0 / conflict 0 / capture_only 0）。
- ir_schema.py 校验通过（69 节点、4 项 kernel 替换、8 个源文件哈希登记）。
