#!/bin/bash
# Clean up test scripts and debug files before GitHub checkin

echo "Cleaning up test and debug files..."
echo ""

cd /scratch/venksrin/project-clustermon

# Remove test scripts
rm -f test-*.py test-*.sh
rm -f check-*.py
rm -f diagnose.sh
rm -f verify-fixes.sh
rm -f test_auth_script.py

# Remove duplicate/debug documentation
rm -f FIXES_APPLIED.md
rm -f TESTING_GUIDE.md
rm -f TESTING.md
rm -f VERIFICATION_CHECKLIST.md
rm -f QUICK_START.md
rm -f STATUS.md
rm -f CURRENT_STATUS.md

# Remove temporary directories
rm -rf .ssh-temp .ssh-temp-setup ssh-keys

# List what's being kept
echo "Files kept for GitHub:"
echo ""
echo "Core application:"
ls -1 | grep -E "^(backend|frontend|cvs|config|scripts)$"

echo ""
echo "Docker files:"
ls -1 | grep -E "Dockerfile|docker-compose.yml|.dockerignore"

echo ""
echo "Configuration:"
ls -1 | grep -E ".gitignore|.env.example"

echo ""
echo "Scripts:"
ls -1 *.sh 2>/dev/null | grep -v cleanup

echo ""
echo "Documentation:"
ls -1 *.md 2>/dev/null

echo ""
echo "✓ Cleanup complete!"
echo ""
echo "Ready for GitHub:"
echo "  git init"
echo "  git add ."
echo "  git commit -m 'Initial commit: Production-ready CVS Cluster Monitor'"
echo ""
