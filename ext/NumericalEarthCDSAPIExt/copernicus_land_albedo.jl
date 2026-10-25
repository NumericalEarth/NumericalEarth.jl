#####
##### Copernicus land surface albedo: this backend only supplies `retrieve`; the request
##### construction, extraction, and repacking live in the CopernicusLandAlbedo module.
#####

Downloads.download(metadata::CopernicusAlbedoDatasetMetadata; retrieve=CDSAPI.retrieve, kwargs...) =
    download_ten_day_albedo!(metadata; kwargs...) do request, path
        retrieve_with_retries(ALBEDO_CDS_PRODUCT, request, path; retrieve)
    end
