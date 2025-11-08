import h5py
import shutil

def repack_h5(src_path, dst_path):
    with h5py.File(src_path, "r") as fsrc, h5py.File(dst_path, "w") as fdst:
        def copy_group(src_grp, dst_grp):
            for key, item in src_grp.items():
                if isinstance(item, h5py.Group):
                    new_group = dst_grp.create_group(key)
                    copy_group(item, new_group)
                else:
                    dst_grp.create_dataset(
                        key,
                        data=item[()],
                        compression=item.compression or "gzip",
                        compression_opts=item.compression_opts or 4,
                    )

        copy_group(fsrc, fdst)

    print(f"✅ Repacked file saved to: {dst_path}")

# Example usage
repack_h5("table_demo.h5", "table_demo_repacked.h5")
