import { useState, useRef } from 'react'
import ImageUpload from './components/ImageUpload'
import SampleImages from './components/SampleImages';
import Prediction from './components/Prediction';
import type { ImageFile } from './ImageFile';
import './App.css';


function App() {
    const [images, setImages] = useState<ImageFile[]>([]);
    const previewRef = useRef<HTMLDivElement | null>(null);

    return (
        <main>
            <h1>Fire Vision</h1>
            <h2>Smoke, Fire, and Non-fire Image Classifier</h2>
            <p>Upload an image (or batch upload up to 100 images) below, and the model will determine whether there is fire, smoke, or if it appears normal. Sample images are available <a href="#sample-images">below</a> — click "Get Prediction" to try them out.</p>
            <ImageUpload setImages={setImages} />

            <Prediction images={images} previewRef={previewRef} />

            <SampleImages setImages={setImages} />
        </main>
    )
}

export default App
