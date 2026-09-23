from PIL import Image

from src.inference import load_trained_model, predict_image
from src.preprocessing import CLASS_NAMES


def test_inference_output():
    model, device = load_trained_model()

    image = Image.new(
        "RGB",
        (224, 224),
        color=(120, 180, 90),
    )

    result = predict_image(
        image,
        model,
        device,
    )

    assert result["prediction"] in CLASS_NAMES

    assert 0.0 <= result["confidence"] <= 1.0

    assert isinstance(result["accepted"], bool)

    assert len(result["top_3"]) == 3

    probabilities = [
        item["confidence"]
        for item in result["top_3"]
    ]

    assert abs(sum(probabilities) - 1.0) < 1e-5
