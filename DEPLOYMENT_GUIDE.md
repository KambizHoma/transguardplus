# TransGuardPlus Deployment Guide 📘

Complete step-by-step instructions for deploying TransGuardPlus to Streamlit Cloud using GitHub Desktop.

---

## Prerequisites

✅ **Before you begin, ensure you have:**

1. **GitHub Account** - Sign up at [github.com](https://github.com) if you don't have one
2. **GitHub Desktop** - Download from [desktop.github.com](https://desktop.github.com)
3. **Streamlit Cloud Account** - Free at [share.streamlit.io](https://share.streamlit.io) (uses GitHub login)
4. **TransGuardPlus Files** - All 5 files ready:
   - `app_streamlit.py`
   - `transguardplus_core.py`
   - `requirements.txt`
   - `sample_config.json`
   - `README.md`

---

## Part 1: Create GitHub Repository with GitHub Desktop

### Step 1.1: Open GitHub Desktop

1. Launch **GitHub Desktop** application
2. Sign in with your GitHub account if prompted
   - Click **Sign in to GitHub.com**
   - Enter credentials in browser
   - Authorize GitHub Desktop

### Step 1.2: Create New Repository

1. Click **File** menu → **New Repository** (or `Ctrl+N` on Windows, `Cmd+N` on Mac)

2. Fill in the repository details:
   ```
   Name: transguardplus
   Description: Advanced Real-Time Bank Transaction Anomaly Platform
   Local path: [Choose where to save - e.g., C:\Users\YourName\Documents\GitHub]
   Initialize this repository with a README: ✅ CHECK THIS BOX
   Git ignore: None
   License: None
   ```

3. Click **Create Repository** button

4. GitHub Desktop will:
   - Create a new folder at your chosen location
   - Initialize a Git repository
   - Create an initial README.md file
   - Make the first commit

### Step 1.3: Add Your Project Files

1. Open the repository folder in File Explorer/Finder:
   - In GitHub Desktop, click **Repository** menu → **Show in Explorer** (Windows)
   - Or **Repository** menu → **Show in Finder** (Mac)

2. You'll see a folder with just `.git` folder and `README.md`

3. Copy/move these 5 files into this folder:
   - `app_streamlit.py`
   - `transguardplus_core.py`
   - `requirements.txt`
   - `sample_config.json`
   - `README.md` (replace the existing one)

4. Return to GitHub Desktop - you'll see all files listed under "Changes" tab

### Step 1.4: Commit Changes

1. In the bottom-left panel, you'll see:
   - **Summary** field (required)
   - **Description** field (optional)

2. Enter commit message:
   ```
   Summary: Initial commit - TransGuardPlus v2.0
   Description: Complete application with animated temporal analysis
   ```

3. Click **Commit to main** button

4. All files are now committed to your local repository (on your computer only)

### Step 1.5: Publish to GitHub

1. Click the **Publish repository** button at the top

2. In the dialog box, verify/update:
   ```
   Name: transguardplus
   Description: Advanced Real-Time Bank Transaction Anomaly Platform with Temporal Pattern Analysis
   Keep this code private: ☐ UNCHECKED (make it public)
                          or ✅ CHECKED (if you want it private)
   Organization: [Leave as "None" unless you have one]
   ```

3. Click **Publish repository** button

4. Wait for upload to complete (usually 10-30 seconds)

5. Success! You'll see:
   - The "Publish repository" button changes to "Fetch origin"
   - Your repository is now on GitHub

6. **Verify on GitHub.com:**
   - Click **Repository** menu → **View on GitHub**
   - Your browser opens showing your repository
   - Confirm all 5 files are visible

---

## Part 2: Deploy to Streamlit Cloud

### Step 2.1: Access Streamlit Cloud

1. Open browser and go to: [share.streamlit.io](https://share.streamlit.io)

2. Click **Sign in** (top right)

3. Choose **Continue with GitHub**

4. Authorize Streamlit Cloud to access your GitHub (if first time):
   - Click **Authorize streamlit**
   - Enter GitHub password if prompted

### Step 2.2: Create New App

1. Once logged in, you'll see the Streamlit Cloud dashboard

2. Click **New app** button (or **Create app** if it's your first)

3. You'll see a deployment configuration form with 3 sections:
   - Repository
   - Branch and file
   - App URL

### Step 2.3: Configure Deployment

**Section 1: Repository**
```
Repository: [Select your username]/transguardplus
```
- Click the dropdown
- Find and select `YOUR_GITHUB_USERNAME/transguardplus`

**Section 2: Branch and file**
```
Branch: main
Main file path: app_streamlit.py
```
- Leave branch as `main` (default)
- Type `app_streamlit.py` in the file path field

**Section 3: App URL (Advanced Settings - Optional)**
```
App URL: transguardplus-demo
```
- Click "Advanced settings" to expand (optional)
- Customize your app's URL slug if desired
- Final URL will be: `https://transguardplus-demo.streamlit.app`
- If you skip this, Streamlit auto-generates a URL

### Step 2.4: Deploy Application

1. Review your settings:
   ```
   ✅ Repository: YOUR_USERNAME/transguardplus
   ✅ Branch: main
   ✅ Main file path: app_streamlit.py
   ```

2. Click **Deploy!** button

3. Deployment process begins:
   - You'll see a **"Your app is being deployed"** screen
   - Progress indicator shows installation steps
   - This takes about 2-3 minutes

4. **Deployment Steps** (you'll see in console):
   ```
   [1/3] Cloning repository...
   [2/3] Installing dependencies from requirements.txt...
   [3/3] Starting application...
   ```

5. **Success!** When deployment completes:
   - The app automatically loads in your browser
   - You'll see the TransGuardPlus interface
   - URL is shown at the top

### Step 2.5: Test Your Deployed App

1. **Test Real-Time Monitor:**
   - Click **▶️ Start Stream** button in sidebar
   - Verify transactions appear
   - Check that KDE plot updates
   - Confirm alert table populates

2. **Test Temporal Analysis:**
   - Click **Temporal Pattern Analysis** tab
   - Wait for data to accumulate (need 10+ transactions)
   - Verify animated bubble chart appears
   - Click **▶ Play** button on the chart
   - Watch animation progress through hours

3. **Test Controls:**
   - Try **⏸️ Stop Stream**
   - Try **⏭️ Step Once**
   - Adjust **Time Window** slider
   - Adjust **Alert Threshold** slider
   - Try **🗑️ Clear Data**

### Step 2.6: Get Your App URL

Your app is now live at one of these URLs:

**Option A - Custom URL (if you set one):**
```
https://YOUR-CUSTOM-NAME.streamlit.app
```

**Option B - Auto-generated URL:**
```
https://YOUR-GITHUB-USERNAME-transguardplus-app-streamlit-RANDOM.streamlit.app
```

**To find your URL:**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in
3. Find your app in the dashboard
4. Click to view - URL is in browser address bar

---

## Part 3: Updating Your App (After Initial Deployment)

### When you make changes to your code:

#### Using GitHub Desktop:

1. **Edit your files** locally (in the repository folder)

2. **Open GitHub Desktop**
   - You'll see changed files under "Changes" tab

3. **Commit changes:**
   ```
   Summary: Update feature X
   Description: Added Y functionality
   ```

4. **Click "Commit to main"**

5. **Push to GitHub:**
   - Click **Push origin** button (top right)

6. **Streamlit auto-deploys:**
   - Within 1-2 minutes, Streamlit Cloud detects the change
   - Automatically rebuilds and redeploys your app
   - No manual action needed!

#### Monitoring Updates:

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click on your app
3. Click **Manage app** (bottom right)
4. View deployment logs in real-time

---

## Part 4: Managing Your App

### App Settings (in Streamlit Cloud Dashboard)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Find your app
3. Click **⋮** (three dots) → **Settings**

**Available options:**
- **General**: Change app name, URL
- **Sharing**: Make app public/private
- **Secrets**: Add environment variables (if needed)
- **Resources**: View usage stats

### Viewing Logs

1. Click **⋮** → **Logs**
2. See real-time application logs
3. Useful for debugging

### Rebooting App

If app becomes unresponsive:
1. Click **⋮** → **Reboot app**
2. Waits ~30 seconds
3. App restarts fresh

### Deleting App

1. Click **⋮** → **Delete app**
2. Confirm deletion
3. App is removed (repository remains on GitHub)

---

## Troubleshooting

### Issue: "App failed to deploy"

**Solution:**
1. Check **Logs** in Streamlit Cloud dashboard
2. Common causes:
   - Missing dependencies in `requirements.txt`
   - Syntax errors in Python code
   - Wrong main file path

### Issue: "Module not found" error

**Solution:**
1. Verify `requirements.txt` includes all packages
2. Check spelling of package names
3. Ensure package versions are compatible

### Issue: Animation not appearing

**Solution:**
1. Need at least 10 transactions for animation
2. Click **⏭️ Step Once** multiple times
3. Or click **▶️ Start Stream** and wait

### Issue: App is slow

**Solution:**
1. Reduce **Time Window** to 60-120 seconds
2. Decrease `tx_per_second` in `sample_config.json`
3. Streamlit Community Cloud has resource limits

---

## Best Practices

### Development Workflow

1. **Test locally first:**
   ```bash
   streamlit run app_streamlit.py
   ```

2. **Commit working code only:**
   - Test thoroughly before pushing
   - Write descriptive commit messages

3. **Use branches for major changes:**
   - Create feature branch in GitHub Desktop
   - Test before merging to main

### Repository Management

1. **Keep README updated:**
   - Document new features
   - Update version numbers

2. **Use .gitignore:**
   - Exclude `data/` folder
   - Exclude `.streamlit/` folder
   - Exclude `__pycache__/`

3. **Tag releases:**
   - Use GitHub releases for versions
   - Tag as v2.0, v2.1, etc.

---

## Quick Reference

### GitHub Desktop Shortcuts

| Action | Windows/Linux | Mac |
|--------|---------------|-----|
| New Repository | Ctrl+N | Cmd+N |
| Commit | Ctrl+Enter | Cmd+Enter |
| Push | Ctrl+P | Cmd+P |
| Pull | Ctrl+Shift+P | Cmd+Shift+P |
| Show in Explorer/Finder | Ctrl+Shift+F | Cmd+Shift+F |

### Streamlit Cloud URLs

- **Dashboard**: [share.streamlit.io](https://share.streamlit.io)
- **Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community**: [discuss.streamlit.io](https://discuss.streamlit.io)

---

## Summary Checklist

### ✅ Pre-Deployment
- [ ] All 5 files prepared
- [ ] GitHub account created
- [ ] GitHub Desktop installed
- [ ] Code tested locally

### ✅ GitHub Setup
- [ ] Repository created in GitHub Desktop
- [ ] Files added to repository
- [ ] Changes committed
- [ ] Repository published to GitHub
- [ ] Verified files on GitHub.com

### ✅ Streamlit Deployment
- [ ] Logged into Streamlit Cloud
- [ ] New app created
- [ ] Repository selected
- [ ] Main file path set to `app_streamlit.py`
- [ ] App deployed successfully
- [ ] App tested and working
- [ ] URL bookmarked

---

## Support Resources

**GitHub Desktop Help:**
- [Desktop Documentation](https://docs.github.com/en/desktop)
- [Getting Started Guide](https://docs.github.com/en/desktop/installing-and-configuring-github-desktop)

**Streamlit Cloud Help:**
- [Deploy an app](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [Manage your app](https://docs.streamlit.io/streamlit-community-cloud/manage-your-app)

**TransGuardPlus Support:**
- Email: nippofin@nippotica.jp

---

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Created for:** TransGuardPlus v2.0 Deployment
