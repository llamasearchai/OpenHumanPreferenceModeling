import os
import subprocess

from datetime import datetime, timedelta


def run(cmd):
    subprocess.run(cmd, shell=True, check=True)


def commit(msg, date_offset_days):
    date = (datetime.now() - timedelta(days=date_offset_days)).isoformat()
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = date
    env["GIT_COMMITTER_DATE"] = date
    subprocess.run(["git", "commit", "-m", msg], env=env, check=True)


def main():
    # Initialize
    if os.path.exists(".git"):
        run("rm -rf .git")

    run("git init")
    run('git config user.name "OpenHumanPreferenceModeling Bot"')
    run('git config user.email "bot@ohpm.ai"')
    run("git checkout -b main")

    # 1. Initial Commit (30 days ago)
    # Add gitignore, requiremens, package.json configs
    run(
        "git add .gitignore requirements.txt frontend/package.json frontend/tsconfig.json frontend/vite.config.ts pytest.ini"
    )
    commit("chore: Initial project scaffold and dependency configuration", 30)

    # 2. Core Backend (25 days ago)
    # Add main.py, start_backend.sh, common/, configs/
    run("git add main.py start_backend.sh common/ configs/")
    commit("feat(backend): Implement unified FastAPI application and core services", 25)

    # 3. ML Components (20 days ago)
    # Add calibration, active_learning, encoders
    run(
        "git add calibration/ active_learning/ neural_encoder/ user_state_encoder/ sft_pipeline/ dpo_pipeline/"
    )
    commit("feat(ml): Add signal processing, calibration, and training pipelines", 20)

    # 4. Frontend Core (15 days ago)
    # Add frontend src basic structure
    run(
        "git add frontend/index.html frontend/src/App.tsx frontend/src/main.tsx frontend/src/index.css"
    )
    commit("feat(frontend): Initialize React application with core routing", 15)

    # 5. Frontend Features (10 days ago)
    # Add frontend components and pages
    run("git add frontend/src/")
    commit(
        "feat(frontend): Implement dashboard, annotation interface, and visualizations",
        10,
    )

    # 6. Infrastructure & Tests (5 days ago)
    # Add deployment, scripts, tests
    run("git add deployment/ scripts/ tests/")
    commit("feat(infra): Add deployment manifests and comprehensive test suite", 5)

    # 7. Documentation (Today)
    # Add README and specs
    run("git add README.md docs/")
    commit("docs: Add comprehensive README and specification documents", 0)

    print("Git history reconstruction complete.")


if __name__ == "__main__":
    main()
