import { useEffect, useState } from "react";
import loadingIcon from "../assets/loading.gif";
import fireIcon from "../assets/fire.png";
import smokeIcon from "../assets/smoke.png";
import checkmarkIcon from "../assets/check.png";
import type { ImageFile } from "../ImageFile";

const apiUrl = import.meta.env.VITE_API_URL;

type Props = {
    image: ImageFile | null;
};

type Prediction = {
    predicted_class: string;
    predicted_class_display: string;
    confidence: number;
    probabilities: Record<string, number>;
}

function Result({ image }: Props) {
    const [prediction, setPrediction] = useState<Prediction| null>(null);
    const [error, setError] = useState<boolean>(false);
    const [loading, setLoading] = useState<boolean>(false);

    const getPrediction = async (file: File) => {
        setPrediction(null);
        setLoading(true);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch(`${apiUrl}/predict`, {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            setPrediction(data);
            setLoading(false);
        } catch (error) {
            console.log(error);
            setError(true);
        }
    };

    const getEmoji = (predictedClass: string) => {
        switch (predictedClass.toLowerCase()) {
            case "fire":
                return <img src={fireIcon} alt="Fire Icon" width="40" />;
            case "smoke":
                return <img src={smokeIcon} alt="Smoke Icon" width="40" />;
            case "non_fire":
                return <img src={checkmarkIcon} alt="Checkmark Icon" width="40" />;
            default:
                return "";
        }
    };

    useEffect(() => {
        if (!image) return;

        URL.revokeObjectURL(image.preview);

        const fetchPrediction = async () => {
            await getPrediction(image.file);
        };

        fetchPrediction();
    }, [image]);

    return (
        <div>
            <div className="preview-result">
                {image && (
                    <div className="preview">
                        <img
                            src={image.preview}
                            alt="Preview"
                            onLoad={() => URL.revokeObjectURL(image.preview)}
                        />
                    </div>
                )}
                <div className={`result border ${prediction?.predicted_class.toLowerCase().replace(/\s/g, '')}`}>
                    {error && (
                        <p>Error getting prediction. Please try again.</p>
                    )}
                    {loading && (
                        <img src={loadingIcon} alt="Loading" />
                    )}
                    {prediction && (
                        <>
                            {getEmoji(prediction.predicted_class)}
                            <p className="mb-0 prediction">Prediction: {prediction?.predicted_class_display}</p>
                            <p className="mb-0"><b>Confidence:</b> {(prediction?.confidence * 100).toFixed(2)}%</p>

                            <div className="probabilities">
                                {Object.entries(prediction.probabilities).map(([className, prob]) => (
                                    <div className="probability-bar" key={className}>
                                        <span className="label">{className}</span>
                                        <div className="bar">
                                            <div 
                                                className="bar-filled"
                                                style={{
                                                    width: `${(prob * 100).toFixed(1)}%`
                                                }}
                                            />
                                        </div>
                                        <span className="probability-value">({(prob * 100).toFixed(2)}%)</span>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}

export default Result
