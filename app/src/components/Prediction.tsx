import { useEffect } from "react";
import type { ImageFile } from "../ImageFile";
import BatchResult from "./BatchResult";
import Result from "./Result";

type Props = {
    images: ImageFile[];
    previewRef: React.RefObject<HTMLDivElement | null>;
};

type Prediction = {
    predicted_class: string;
    predicted_class_display: string;
    confidence: number;
    probabilities: Record<string, number>;
}

function Prediction({ images, previewRef }: Props) {
    useEffect(() => {
        if (!previewRef.current) return;

        previewRef.current?.scrollIntoView({
            behavior: 'smooth',
            block: 'nearest',
            inline: 'center'
        });
    }, [images]);

    return (
        <>
            {images.length > 0 && (
                <h2 ref={previewRef}>Prediction</h2>
            )}
            {images.length == 1 && (
                <Result image={images[0]} />
            )}
            {images.length > 1 && (
                <BatchResult images={images} />
            )}
        </>
    )
}

export default Prediction
