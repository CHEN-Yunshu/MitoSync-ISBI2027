"""Create a data/output layout and source links; never download assets or overwrite paths."""
from pathlib import Path
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.expanduser().resolve()
    source=Path(__file__).resolve().parents[1]/'code'
    # Verify existing links before making changes.
    for name in ('vace','mitosync'):
        target=root/'scripts'/name
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve()!=source/name:
                raise SystemExit(f'Refusing to replace existing path: {target}')
    for folder in ('data','eval','ext','scripts'):(root/folder).mkdir(parents=True,exist_ok=True)
    for name in ('vace','mitosync'):
        target=root/'scripts'/name
        if not target.is_symlink():target.symlink_to(source/name,target_is_directory=True)
    print(f'Workspace ready: {root}')
    print('Set MITOSYNC_ROOT and N to this path; supply data and external model weights separately.')

if __name__=='__main__':main()
