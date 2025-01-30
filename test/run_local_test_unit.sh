#!/bin/bash
pytest --cov=ard --cov=gpAEP --cov-report=html test/unit

rm -rf test/unit/layout/problem*_out

#
