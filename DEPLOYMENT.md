# 🚀 Deployment Guide for Streamlit Cloud

This guide will help you deploy your Options Profit Calculator to Streamlit Cloud (\*.streamlit.app).

## Prerequisites

1. A GitHub account
2. A Google API Key (free from Google AI Studio)

## Step-by-Step Deployment

### Step 1: Get Your Google API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key (keep it safe!)

### Step 2: Push Code to GitHub

1. **Create a new GitHub repository:**

   - Go to [github.com/new](https://github.com/new)
   - Name it something like `options-calculator`
   - Choose Public or Private
   - Don't initialize with README (we already have one)

2. **Push your code:**
   ```bash
   cd d:\options
   git init
   git add .
   git commit -m "Initial commit - Options Calculator"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/options-calculator.git
   git push -u origin main
   ```

### Step 3: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Configure your app:**

   - **Repository:** Select your `options-calculator` repo
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** Choose your custom URL (e.g., `my-options-calc`)

5. **Add Secrets:**

   - Click "Advanced settings"
   - In the "Secrets" section, paste:

   ```toml
   GOOGLE_API_KEY = "your-actual-api-key-here"
   ```

   - Replace `your-actual-api-key-here` with your actual Google API key

6. **Click "Deploy"**

Your app will be live at: `https://your-app-name.streamlit.app`

## Step 4: Test Your Deployment

1. Visit your app URL
2. Select a strategy (e.g., "Long Call")
3. Click "Generate Comprehensive Report"
4. Verify the AI analysis appears

## Troubleshooting

### Common Issues

**Problem:** "Module not found" error

- **Solution:** Make sure `requirements.txt` is in your repository root
- Check that all dependencies are listed

**Problem:** AI analysis shows "API key not configured"

- **Solution:** Double-check your secrets configuration in Streamlit Cloud
- Make sure the key is named exactly `GOOGLE_API_KEY`
- No quotes around the value in secrets

**Problem:** App crashes on startup

- **Solution:** Check the logs in Streamlit Cloud dashboard
- Verify Python version compatibility (3.8+)

**Problem:** Charts not displaying

- **Solution:** Make sure `plotly` is in requirements.txt
- Clear browser cache and reload

## Updating Your App

After deployment, any changes you push to GitHub will automatically trigger a redeploy:

```bash
# Make your changes, then:
git add .
git commit -m "Description of changes"
git push
```

Streamlit Cloud will automatically detect the changes and redeploy (usually takes 1-2 minutes).

## Advanced Configuration

### Custom Domain

1. In Streamlit Cloud dashboard, go to your app settings
2. Click "Custom subdomain"
3. Choose your preferred name

### Resource Limits

Free tier includes:

- 1 GB RAM
- 1 CPU core
- Suitable for moderate traffic

For higher traffic, consider:

- Streamlit Cloud Teams ($250/month)
- Self-hosting on AWS/GCP/Azure

### Monitoring

- **Usage stats:** Available in Streamlit Cloud dashboard
- **Logs:** Click "Manage app" → "Logs"
- **Analytics:** Set up Google Analytics (optional)

## Security Best Practices

1. **Never commit secrets:**

   - Always use `.gitignore` to exclude `secrets.toml`
   - Use Streamlit Cloud secrets management

2. **API Key protection:**

   - Keep your Google API key private
   - Regenerate if accidentally exposed

3. **Rate limiting:**
   - Be aware of Google API quotas
   - Implement caching for repeated requests

## Cost Considerations

### Free Tier

- **Streamlit Cloud:** Free for public apps
- **Google Gemini API:** Free tier includes 60 requests/minute

### Paid Options

- **Streamlit Cloud Teams:** $250/month for private apps
- **Google Gemini Pro:** $0.00025 per 1K characters

## Support

- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum:** [discuss.streamlit.io](https://discuss.streamlit.io)
- **Google AI Docs:** [ai.google.dev](https://ai.google.dev)

---

## Quick Deployment Checklist

- [ ] Google API key obtained
- [ ] Code pushed to GitHub
- [ ] Streamlit Cloud account created
- [ ] App configured with correct repo/branch/file
- [ ] Secrets added to Streamlit Cloud
- [ ] App deployed successfully
- [ ] AI analysis tested and working
- [ ] Shared app URL with users

**Congratulations! Your options calculator is now live! 🎉**
