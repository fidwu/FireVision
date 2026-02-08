import { useEffect, useState } from 'react';
import type { ImageFile } from '../ImageFile';
import loadingIcon from "../assets/loading.gif";

const apiUrl = import.meta.env.VITE_API_URL;

type Props = {
    images: ImageFile[];
};

type Prediction = {
    filename: string;
    predicted_class: string;
    predicted_class_display: string;
    confidence: number;
    probabilities: Record<string, number>;
}

type BatchPrediction = {
    results: Prediction[];
};

function BatchResult({ images }: Props) {
    const [batchPrediction, setBatchPrediction] = useState<BatchPrediction | null>(null);
    const [error, setError] = useState<boolean>(false);
    const [loading, setLoading] = useState<boolean>(false);
    const [numImages, setNumImages] = useState<number>(0);
    
    const getPrediction = async (files: ImageFile[]) => {
        setError(false);
        setBatchPrediction(null);
        setLoading(true);

        try {
            const formData = new FormData();
            files.forEach(file => {
                formData.append("files", file.file, file.file.name);
            });

            const response = await fetch(`${apiUrl}/predict/batch`, {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            setBatchPrediction(data);
            setLoading(false);
        } catch (error) {
            console.error(error);
            setError(true);
            setLoading(false);
        }
    };

    useEffect(() => {
        if (!images || images.length === 0) return;

        const updateStateAndPredict = async () => {
            setNumImages(images.length);
            await getPrediction(images);
        };

        updateStateAndPredict();
    }, [images]);

    return (
        <>
            {error && (
                <p>Error getting prediction. Please try again.</p>
            )}
            {loading && (
                <>
                    <img src={loadingIcon} alt="Loading" />
                    <p>Processing {numImages} images</p>
                </>
            )}
            {batchPrediction && (
                <div className="sample-images-grid">
                    {batchPrediction?.results.map((pred, i) => {
                        const image = images[i];
                        return (
                            <div className="sample-image" key={pred.filename}>
                                {image && (
                                    <img
                                        src={image.preview}
                                        alt={pred.filename}
                                        style={{
                                            width: 100,
                                            height: 100,
                                            objectFit: 'cover',
                                            borderRadius: 8
                                        }}
                                    />
                                )}
                                <div className={`result ${pred.predicted_class}`}>
                                    <p className="mb-0 prediction">Prediction: {pred.predicted_class_display}</p>
                                    <p className="mb-0"><b>Confidence:</b> {(pred.confidence * 100).toFixed(2)}%</p>
                                    <div className="probabilities">
                                        {Object.entries(pred.probabilities).map(([className, prob]) => (
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
                                </div>
                            </div>
                        )
                    })}
                </div>
            )}
        </>
    )
}

export default BatchResult
