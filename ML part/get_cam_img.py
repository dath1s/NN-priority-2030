from hikvisionapi import Client


def getCamImg(filename: str) -> None:
    try:
        cam = Client("http://10.50.16.34/", "admin", "DngnM4ster!")

        response = cam.Streaming.channels[102].picture(method="get", type="opaque_data")
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    except Exception:
        print("Something went wrong")


# getCamImg("images/get/img.jpg")
