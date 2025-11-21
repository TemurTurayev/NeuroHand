# CLAUDE.md - Developer & Medical Student Configuration

## About Developer
- **Profile**: 5th year medical student at TashPMI, future pediatrician with bioengineering focus
- **Experience Level**: Beginner in coding, interested in no-code solutions
- **Primary Language**: Python (learning C#)
- **Editor**: Claude Code on MacBook air M4
- **Academic Performance**: 85% average
- **Languages**: English, Russian, Uzbek

## Core Principles
- **НИКОГДА НЕ СДАВАЙСЯ**: При неудаче пробуй минимум 3 разных подхода
- **Подробные объяснения**: Всегда предоставлять логи ошибок и конкретные описания проблем
- **Активные запросы**: Запрашивать дополнительную информацию для полного решения
- **Медицинский контекст**: Учитывать медицинскую специализацию при технических решениях

## Code Style Guidelines

### Python
- Use descriptive variable names (especially for medical data)
- Prefer type hints for function parameters and returns
- Use docstrings for functions handling medical/healthcare data
- Import structure: standard library → third-party → local imports
- Use virtual environments (mention when setting up projects)

### General
- Comment complex medical calculations or algorithms
- Use meaningful commit messages
- Prefer readable code over clever code
- Test medical/healthcare related functions thoroughly

## Common Commands

### Python Development
```bash
# Virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Code quality
python -m flake8 .
python -m black .
python -m pytest

# Package management
pip freeze > requirements.txt
```

### Git Workflow
```bash
git status
git add .
git commit -m "feat: descriptive message"
git push origin main
```

## Medical/Healthcare AI Context
- **Data Privacy**: Always mention HIPAA/medical data privacy considerations
- **Validation**: Emphasize testing and validation for healthcare applications
- **Documentation**: Medical applications require extensive documentation
- **Ethical Considerations**: Highlight AI ethics in healthcare contexts

## Repository Structure Preferences
```
project/
├── src/
│   ├── data/          # Medical datasets (anonymized)
│   ├── models/        # AI/ML models
│   ├── utils/         # Utility functions
│   └── tests/         # Unit tests
├── docs/              # Documentation
├── requirements.txt   # Dependencies
├── README.md         # Project overview
└── CLAUDE.md         # This file
```

## Troubleshooting Approach
1. **First Attempt**: Direct solution with explanation
2. **Second Attempt**: Alternative approach if first fails
3. **Third Attempt**: Simplified or no-code solution if applicable
4. **Always**: Provide detailed error logs and next steps

## Special Instructions
- **GitHub Repositories**: Warn about message limits, continue in next request
- **Medical Context**: Apply medical knowledge when relevant to coding problems
- **Learning Focus**: Explain concepts clearly for beginner-level understanding
- **Language**: Respond in the language of the question (English/Russian/Uzbek)

## AI/ML Healthcare Considerations
- Data preprocessing for medical datasets
- Model interpretability for clinical decisions
- Regulatory compliance (FDA, medical device standards)
- Integration with hospital systems (HL7, FHIR standards)

## Contact & Resources
- **GitHub**: TemurTurayev
- **Email**: temurturayev7822@gmail.com
- **Telegram**: @Turayev_Temur
- **LinkedIn**: linkedin.com/in/temur-turaev-389bab27b/

## Project Types to Prioritize
- Healthcare AI applications
- Medical data analysis tools
- Pediatric care solutions
- Bioengineering projects
- Educational medical software

---
*Last updated: Customize as needed for specific projects*

## 🔒 Security Best Practices

**NEVER commit tokens or secrets to Git!**

For GitHub authentication:
1. Create `.env` file in project root (already in .gitignore)
2. Add: `GITHUB_TOKEN=your_token_here`
3. Use in Python: `import os; token = os.getenv('GITHUB_TOKEN')`
4. For git commands: `git config credential.helper store`

**⚠️ IMPORTANT**: If you accidentally committed a token:
1. Revoke it immediately at: https://github.com/settings/tokens
2. Generate new token
3. Store securely in `.env` file