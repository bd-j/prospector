# syntax=docker/dockerfile:1
#
# Prospector runtime image, for running SED fits on clusters.
#
# python-fsps reads its isochrones, stellar libraries and dust models from
# $SPS_HOME at run time. FSPS_COMMIT is the FSPS commit the pinned fsps wheel was
# compiled from (python-fsps's src/fsps/libfsps submodule at that tag), so the
# data matches the code; update the two together.
FROM python:3.12-slim-bookworm

ARG FSPS_VERSION=0.5.0
ARG FSPS_COMMIT=05b5e550ddd7ffb1e71102b75cfbfdf057c211fd

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    SPS_HOME=/opt/fsps

# setuptools_scm also needs git to version the install from this checkout.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git init -q "$SPS_HOME" \
    && git -C "$SPS_HOME" fetch -q --depth 1 https://github.com/cconroy20/fsps.git "$FSPS_COMMIT" \
    && git -C "$SPS_HOME" checkout -q FETCH_HEAD \
    && rm -rf "$SPS_HOME/.git"

WORKDIR /src
COPY . .

RUN python -m pip install --upgrade pip \
    && pip install "fsps==${FSPS_VERSION}" .

# Fail the build unless FSPS can compute a model SED from the data in $SPS_HOME.
RUN python - <<'EOF'
import numpy as np
from prospect.models import SpecModel, templates
from prospect.observation import Photometry
from prospect.sources import CSPSpecBasis

sps = CSPSpecBasis(zcontinuous=1)
model = SpecModel(templates.TemplateLibrary["parametric_sfh"])
obs = Photometry(filters=[f"sdss_{b}0" for b in "ugriz"],
                 flux=np.ones(5), uncertainty=np.ones(5) / 10)
obs.rectify()
(phot,), _ = model.predict(model.theta, observations=[obs], sps=sps)
assert np.all(np.isfinite(phot)) and np.all(phot > 0), phot
print("prospector runtime OK, ugriz maggies:", phot)
EOF

CMD ["python"]
