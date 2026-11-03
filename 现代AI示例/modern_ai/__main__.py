"""全部演示共用一个环境和一个命令入口。"""
import argparse
import importlib
import json
import sys


def main():
    # 固定 UTF-8 输出，避免中文路径和资料在 Windows 终端中失真。
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('demo', choices=['math', 'learning', 'networks', 'language', 'retrieval'])
    args = parser.parse_args()
    module = importlib.import_module('.'+{'math':'mathlab'}.get(args.demo, args.demo), 'modern_ai')
    action = module.experiment if args.demo == 'learning' else module.demo
    print(json.dumps(action(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
