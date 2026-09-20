"""
Fixes Qwen3.5 GGUF metadata that references a "phantom" MTP (next-token-
prediction) block whose weights were never actually exported.

Qwen3.5's config reports block_count = N (real transformer blocks) + 1 for
the MTP head, but the conversion only exported the N real blocks. This
leaves 3 metadata fields inconsistent with what's actually in the file:

  - qwen35.block_count                  : N+1 -> N
  - qwen35.nextn_predict_layers         : 1   -> 0
  - qwen35.attention.recurrent_layers   : array of N+1 bools -> first N only

This copies every tensor and every other metadata field unchanged, and only
rewrites those three.

Usage:
    uv run python fix_qwen35_gguf.py school-assistant-q4_k_m.gguf school-assistant-q4_k_m-fixed.gguf
"""

import sys
import gguf

def main():
    if len(sys.argv) != 3:
        print("Usage: python fix_qwen35_gguf.py <input.gguf> <output.gguf>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    reader = gguf.GGUFReader(input_path)

    arch_field = reader.get_field("general.architecture")
    if arch_field is None:
        print("ERROR: could not find general.architecture in the input file")
        sys.exit(1)
    arch = arch_field.contents()
    print(f"Architecture: {arch}")

    block_count_key = f"{arch}.block_count"
    nextn_key = f"{arch}.nextn_predict_layers"
    recurrent_key = f"{arch}.attention.recurrent_layers"

    block_count_field = reader.get_field(block_count_key)
    if block_count_field is None:
        print(f"ERROR: {block_count_key} not found -- is this really a qwen35 model?")
        sys.exit(1)

    old_block_count = block_count_field.contents()
    new_block_count = old_block_count - 1
    print(f"{block_count_key}: {old_block_count} -> {new_block_count}")

    writer = gguf.GGUFWriter(output_path, arch=arch)

    for field in reader.fields.values():
        name = field.name
        if name == "general.architecture" or name.startswith("GGUF."):
            continue  # handled by GGUFWriter(arch=...) already

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()

        if name == block_count_key:
            value = new_block_count
            print(f"  patched {name} -> {value}")
        elif name == nextn_key:
            value = 0
            print(f"  patched {name} -> {value}")
        elif name == recurrent_key:
            old_len = len(value)
            value = value[:new_block_count]
            print(f"  patched {name}: {old_len} entries -> {len(value)} entries")

        writer.add_key_value(name, value, val_type, sub_type=sub_type)

    print("Copying tensors...")
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print(f"\nDone. Patched file written to: {output_path}")


if __name__ == "__main__":
    main()
