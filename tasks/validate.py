"""任务集校验入口：python -m tasks.validate
对 tasks/v1 下全部任务执行双向自检（空轨迹必 fail、GT 路径必 pass）。
"""
from tasks.verifiers import main

if __name__ == "__main__":
    raise SystemExit(main())
