import type { ImageFile } from '../ImageFile';
import test_fire_1 from '../assets/test_fire_1.jpg';
import test_fire_2 from '../assets/test_fire_2.jpg';
import test_smoke_1 from '../assets/test_smoke_1.jpg';
import test_smoke_2 from '../assets/test_smoke_2.jpg';
import test_normal_1 from '../assets/test_normal_1.jpg';
import test_normal_2 from '../assets/test_normal_2.jpg';

type Props = {
    setImages: React.Dispatch<React.SetStateAction<ImageFile[]>>;
};

function SampleImages({ setImages }: Props) {
    const selectSample = async (imgPath: string) => {
        const response = await fetch(imgPath);
        const blob = await response.blob();
        const file = new File([blob], "sample.jpg", { type: blob.type });
        setImages([
            {
                file,
                preview: URL.createObjectURL(blob),
            }
        ]);
    };

    const images = [test_fire_1, test_fire_2, test_smoke_1, test_smoke_2, test_normal_1, test_normal_2];

    return (
        <>
            <h2 id="sample-images">Sample Images</h2>
            <div className="sample-images-grid">
                {images.map((image, index) => 
                    <div className="sample-image" key={index}>
                        <img src={image} alt={`Sample ${index + 1}`} width={175} />
                        <button className="link-button" onClick={() => selectSample(image)}>
                            Get prediction
                        </button>
                    </div>
                )}
            </div>
        </>
    )
}

export default SampleImages
