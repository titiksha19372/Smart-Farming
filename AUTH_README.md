# Authentication System

## Overview
The Plant Disease Detection System now includes a complete authentication system with:
- Home page with navigation
- User registration
- User login
- Protected disease detection feature

## How to Use

### 1. Start the Application
```bash
streamlit run app.py
```

### 2. Navigation Flow
1. **Home Page**: Landing page with Login and Register buttons
2. **Register**: Create a new account with username, email, and password
3. **Login**: Sign in with your credentials
4. **Disease Detection**: Upload plant images and get AI-powered analysis (requires login)

### 3. Features
- **Secure Authentication**: Passwords are hashed using SHA-256
- **SQLite Database**: User data stored locally in `users.db`
- **Session Management**: Stay logged in during your session
- **Protected Routes**: Disease detection only accessible after login
- **User-Friendly Navigation**: Easy navigation between pages

## File Structure
```
smart farming/
├── app.py              # Main application with all pages
├── auth.py             # Authentication functions
├── users.db            # SQLite database (created automatically)
├── utils/
│   ├── model_loader.py
│   └── image_processor.py
└── pages/              # (Optional - not used in current implementation)
```

## Security Notes
- Passwords are hashed before storage
- No plain text passwords are stored
- Each user has a unique username
- Session state manages authentication

## Troubleshooting
- If you get a database error, delete `users.db` and restart the app
- Make sure all dependencies are installed: `streamlit`, `sqlite3`, `hashlib`
- The model files must be present for disease detection to work
