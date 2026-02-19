---
sidebar_position: 4
title: Serialization
---

# Model Serialization

Save and load trained models with Aicraft's binary serialization format.

## Save a Model

```c
ac_param_group params;
ac_param_group_init(&params);
ac_param_group_add(&params, fc1.weight);
ac_param_group_add(&params, fc1.bias);
ac_param_group_add(&params, fc2.weight);
ac_param_group_add(&params, fc2.bias);

ac_model_save("model.acml", &params);
```

## Load a Model

```c
ac_error_code err = ac_model_load("model.acml", &params);
if (err != AC_OK) {
    printf("Load failed: %s\n", ac_get_last_error_message());
}
```

:::caution Parameter Order
Parameters must be added to the `ac_param_group` in the **same order** for both save and load. The format stores tensors sequentially without names.
:::

## File Format (`.acml`)

The `.acml` format uses a simple versioned binary layout:

```
┌──────────────────────┐
│  Magic: "ACML"       │  4 bytes — corruption detection
├──────────────────────┤
│  Version: uint32     │  4 bytes — format version
├──────────────────────┤
│  Num params: uint32  │  4 bytes
├──────────────────────┤
│  For each parameter: │
│  ├── ndim: uint32    │  Number of dimensions
│  ├── shape[ndim]     │  Dimension sizes
│  └── data[...]       │  Float32 values (little-endian)
└──────────────────────┘
```

## Error Handling during I/O

```c
ac_error_code err = ac_model_load("missing.acml", &params);
if (err != AC_OK) {
    printf("Error code: %d\n", err);
    printf("Error name: %s\n", ac_error_string(err));
    printf("Message: %s\n", ac_get_last_error_message());
    ac_clear_error();
}
```

Possible errors:
- `AC_ERROR_IO` — File not found or unreadable
- `AC_ERROR_FORMAT` — Invalid magic header or version mismatch
- `AC_ERROR_SHAPE` — Parameter shape mismatch during load
