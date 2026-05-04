#!/bin/bash
# Setup script for Meeting Copilot

echo "Meeting Copilot Setup"
echo "===================="

# Create venv
echo "Creating virtual environment..."
python -m venv venv

# Activate venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Provide next steps
echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Get your HuggingFace token at https://huggingface.co/settings/tokens"
echo "2. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.0"
echo "3. Set the HF_TOKEN environment variable:"
echo "   export HF_TOKEN='your_token_here'"
echo "4. Run the transcriber:"
echo "   python transcribe.py"
